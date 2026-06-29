"""
NLU 意图分类训练脚本
=====================================
数据来源：training_data/ 目录下的 CSV 文件
每个 CSV 文件 = 一个意图分类，文件名就是分类名
CSV 格式：两列 —— input（用户话语）、label（分类名）

用法示例：
  # 自动检测设备（有 GPU 就用 GPU，没有就用 CPU）
  python train_nlu.py

  # 强制用 CPU（自己电脑）
  python train_nlu.py --device cpu

  # 强制用 GPU（服务器）
  python train_nlu.py --device cuda

  # 自定义参数
  python train_nlu.py --device cpu --epochs 15 --batch_size 8
"""

import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)

# 固定随机种子，保证每次训练结果可复现（类似 Android 里固定测试数据）
set_seed(42)

# 关闭 tokenizer 的并行警告（多进程时会有冲突提示，关掉就不烦了）
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# 1. 数据集类
#    类比 Android 的 RecyclerView.Adapter：
#    把原始数据列表包装成模型可以逐条取用的格式
# ─────────────────────────────────────────────
class NLUDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        """
        texts    : 用户话语列表，例如 ["打开空调", "导航到天安门"]
        labels   : 对应的整数标签列表，例如 [0, 3]
        tokenizer: BERT 分词器，负责把中文句子切成 token id
        max_len  : 句子最大长度，超过就截断（车控场景句子都很短，64 够用）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # 返回数据集总条数（Dataset 的必须方法）
        return len(self.texts)

    def __getitem__(self, idx):
        # 每次取一条数据时调用，把文本转成 BERT 需要的 token id 格式
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,       # 超过 max_len 就截断
            padding=False,         # 不在这里补齐，交给 DataCollator 统一处理
            max_length=self.max_len,
            return_tensors=None,   # 返回普通 list，不是 tensor（DataCollator 会转）
        )
        # 把标签塞进去，Trainer 训练时需要
        encoding["label"] = self.labels[idx]
        return encoding


# ─────────────────────────────────────────────
# 2. 加载数据
#    扫描 training_data/ 目录，读取所有 CSV
# ─────────────────────────────────────────────
def load_data(data_dir):
    """
    扫描 data_dir 目录下所有 .csv 文件，合并成 texts + labels 两个列表。
    每个 CSV 的文件名（去掉 .csv）就是意图分类名。
    """
    texts, labels = [], []

    # 找出目录下所有 csv 文件并排序（排序让每次加载顺序一致）
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到任何 CSV 文件")

    print(f"  找到 {len(csv_files)} 个分类文件：")

    for filename in csv_files:
        # 分类名 = 文件名去掉 .csv 后缀
        category = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)

        df = pd.read_csv(filepath, encoding="utf-8-sig")

        # 检查 CSV 格式是否正确
        if "input" not in df.columns:
            print(f"  ⚠ [{category}] 缺少 'input' 列，跳过")
            continue

        # 过滤空行
        valid = df["input"].dropna().astype(str).str.strip()
        valid = valid[valid.str.len() >= 2].tolist()

        if not valid:
            print(f"  ⚠ [{category}] 无有效数据，跳过")
            continue

        texts.extend(valid)
        labels.extend([category] * len(valid))
        print(f"    [{category}] {len(valid)} 条")

    return texts, labels


# ─────────────────────────────────────────────
# 3. 带类别权重的 Trainer（处理数据不均衡）
#    类比：某些意图样本少，训练时给它更高权重，防止模型只偏向样本多的类
# ─────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 把权重张量存下来，compute_loss 时用
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 从输入里取出标签
        labels = inputs.pop("labels")

        # 前向传播，得到 logits（每个类别的原始评分）
        outputs = model(**inputs)
        logits = outputs.logits

        # 用带权重的交叉熵损失（样本少的类别权重大，训练时更受重视）
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# 4. 主函数
# ─────────────────────────────────────────────
def main():
    # ── 解析命令行参数 ──────────────────────────
    parser = argparse.ArgumentParser(description="BERT NLU 意图分类训练")
    parser.add_argument(
        "--data_dir", type=str, default="./training_data",
        help="CSV 训练数据目录（默认：./training_data）"
    )
    parser.add_argument(
        "--model_name", type=str, default="./bert-base-chinese",
        help="预训练 BERT 模型路径（默认：./bert-base-chinese）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./nlu_model",
        help="训练好的模型保存目录（默认：./nlu_model）"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="训练设备：auto=自动检测 / cpu=强制CPU / cuda=强制GPU（默认：auto）"
    )
    parser.add_argument("--max_len",    type=int,   default=64,   help="句子最大 token 长度（默认：64）")
    parser.add_argument("--epochs",     type=int,   default=10,   help="训练轮数（默认：10）")
    parser.add_argument("--batch_size", type=int,   default=16,   help="批大小（CPU 建议 8，GPU 可以 32）")
    parser.add_argument("--lr",         type=float, default=2e-5, help="学习率（默认：2e-5）")
    parser.add_argument("--test_size",  type=float, default=0.2,  help="测试集比例（默认：0.2 即 20%）")
    args = parser.parse_args()

    print("=" * 60)
    print("   BERT NLU 意图分类 · 训练")
    print("=" * 60)

    # ── 确定训练设备 ────────────────────────────
    if args.device == "auto":
        # 自动检测：有英伟达 GPU 就用 CUDA，否则用 CPU
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n设备检测：{'检测到 GPU，使用 CUDA' if device_name == 'cuda' else '未检测到 GPU，使用 CPU'}")
    else:
        device_name = args.device
        if device_name == "cuda" and not torch.cuda.is_available():
            print("⚠ 指定了 cuda 但未检测到 GPU，回退到 CPU")
            device_name = "cpu"

    use_cpu = (device_name == "cpu")
    print(f"训练设备：{device_name.upper()}")

    # ── 第一步：加载数据 ────────────────────────
    print(f"\n[1/5] 加载训练数据：{args.data_dir}")
    texts, labels = load_data(args.data_dir)
    print(f"\n  合计：{len(texts)} 条话语，{len(set(labels))} 个意图分类")

    if len(texts) < 20:
        raise ValueError(f"数据量太少（{len(texts)} 条），请检查 CSV 文件")

    # ── 第二步：标签编码 ────────────────────────
    # 类比 Android：把分类名（String）映射成整数 id（像 ViewType）
    unique_labels = sorted(set(labels))          # 排序后的所有分类名
    label2id = {l: i for i, l in enumerate(unique_labels)}  # 分类名 → 整数
    id2label  = {i: l for l, i in label2id.items()}         # 整数 → 分类名
    num_labels = len(unique_labels)
    label_ids = [label2id[l] for l in labels]    # 把每条数据的标签转成整数

    # 打印各分类样本数
    print(f"\n  意图分布：")
    counter = Counter(labels)
    for l, c in counter.most_common():
        bar = "▓" * (c // 5 + 1)
        print(f"    {l:12s} {c:3d} 条  {bar}")

    # ── 第三步：划分训练集 / 测试集 ─────────────
    # stratify 参数让每个分类在训练集和测试集里的比例保持一致
    # 类比：保证每种意图都有代表出现在测试集里
    can_stratify = all(c >= 2 for c in counter.values())
    if can_stratify:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, label_ids,
            test_size=args.test_size,
            random_state=42,
            stratify=label_ids,    # 按比例分层抽样
        )
    else:
        # 某些分类只有 1 条数据，无法分层，随机打乱后直接切分
        print("  ⚠ 部分分类样本数 < 2，改用随机划分")
        indices = list(range(len(texts)))
        random.shuffle(indices)
        split = int(len(indices) * (1 - args.test_size))
        train_texts  = [texts[i]     for i in indices[:split]]
        test_texts   = [texts[i]     for i in indices[split:]]
        train_labels = [label_ids[i] for i in indices[:split]]
        test_labels  = [label_ids[i] for i in indices[split:]]

    print(f"\n[2/5] 数据划分：训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条")

    # ── 第四步：加载预训练模型 ───────────────────
    print(f"\n[3/5] 加载预训练模型：{args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # BertForSequenceClassification = BERT 主体 + 顶部分类头
    # 类比：BERT 是预训练好的"特征提取引擎"，分类头是我们加的"业务逻辑层"
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 计算类别权重（样本少的分类权重更大，避免模型偏向大类）
    total = sum(counter.values())
    class_weights = torch.tensor(
        [total / max(counter[id2label[i]], 1) for i in range(num_labels)],
        dtype=torch.float32,
    )

    # ── 第五步：训练 ────────────────────────────
    print(f"\n[4/5] 开始训练")
    print(f"  设备={device_name.upper()}  epochs={args.epochs}  "
          f"batch_size={args.batch_size}  lr={args.lr}")

    # 把文本列表包装成 Dataset 对象
    train_dataset = NLUDataset(train_texts, train_labels, tokenizer, args.max_len)
    test_dataset  = NLUDataset(test_texts,  test_labels,  tokenizer, args.max_len)

    # TrainingArguments：训练超参数配置（类比 Android Gradle 的 buildConfig）
    training_args = TrainingArguments(
        output_dir=args.output_dir,                        # 模型 checkpoint 保存路径
        num_train_epochs=args.epochs,                      # 总训练轮数
        per_device_train_batch_size=args.batch_size,       # 每步训练的样本数
        per_device_eval_batch_size=args.batch_size * 2,   # 评估时批次可以大一倍（不需要梯度）
        learning_rate=args.lr,                             # 学习率
        warmup_ratio=0.1,          # 前 10% 的步数用于"预热"，学习率从 0 逐渐升到设定值
        weight_decay=0.01,         # L2 正则化，防止过拟合
        logging_steps=10,          # 每 10 步打印一次训练 loss
        eval_strategy="epoch",     # 每个 epoch 结束后评估一次
        save_strategy="epoch",     # 每个 epoch 结束后保存一次 checkpoint
        save_total_limit=2,        # 最多保留 2 个 checkpoint，节省磁盘
        load_best_model_at_end=True,          # 训练结束后自动加载最优 checkpoint
        metric_for_best_model="eval_f1",      # 用 F1 分数判断哪个 checkpoint 最好
        greater_is_better=True,               # F1 越高越好
        report_to="none",                     # 不上传到 wandb 等远程平台
        remove_unused_columns=False,          # 保留 label 列（自定义 Dataset 需要）
        use_cpu=use_cpu,           # 关键：True = 强制 CPU，False = 使用 GPU
    )

    # 评估指标：准确率 + F1
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)  # 取概率最高的类别
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    # 使用自定义的 WeightedTrainer（带类别权重的损失函数）
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),  # 自动补齐同批次的短句子
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 连续 3 个 epoch 没提升就提前停止
    )

    trainer.train()

    # 保存最终模型和 tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── 第六步：评估 & 报告 ──────────────────────
    print(f"\n[5/5] 评估结果")

    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = predictions.label_ids

    # 打印每个意图的精确率 / 召回率 / F1
    print(f"\n  分类报告：")
    print(classification_report(
        y_true, y_pred,
        target_names=[id2label[i] for i in range(num_labels)],
        digits=4,
    ))

    # 混淆矩阵（行=真实标签，列=预测标签，对角线上的数越大越好）
    cm = confusion_matrix(y_true, y_pred)
    col_headers = "".join(f"{id2label[i][:5]:>7}" for i in range(num_labels))
    print(f"  混淆矩阵（行=真实，列=预测）：")
    print(f"  {'':>7}{col_headers}")
    for i in range(num_labels):
        row = "".join(f"{cm[i][j]:7d}" for j in range(num_labels))
        print(f"  {id2label[i][:5]:>7}{row}")

    # 保存标签映射文件（预测脚本加载模型时需要用到）
    label_mapping_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label},
                  f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 模型已保存：{args.output_dir}")
    print(f"  ✅ 标签映射：{label_mapping_path}")
    print(f"  ✅ 测试集准确率：{accuracy_score(y_true, y_pred):.2%}")

    # 随机抽 8 条测试样本，直观展示预测效果
    print(f"\n  预测示例（随机抽 8 条）：")
    sample_indices = random.sample(range(len(test_texts)), min(8, len(test_texts)))
    for i in sample_indices:
        true_label = id2label[y_true[i]]
        pred_label = id2label[int(y_pred[i])]
        mark = "✓" if true_label == pred_label else "✗"
        print(f"  [{mark}] {test_texts[i][:50]}")
        print(f"       真实={true_label}  预测={pred_label}")


if __name__ == "__main__":
    main()
