"""
NLU 意图分类预测脚本
=====================================
加载训练好的模型，对用户输入的话语进行意图分类。

用法示例：
  # 单条预测
  python predict_nlu.py --text "打开空调"

  # 多条预测
  python predict_nlu.py --text "打开空调" --text "导航到天安门"

  # 交互模式（连续输入，适合测试）
  python predict_nlu.py --interactive

  # 指定设备
  python predict_nlu.py --interactive --device cpu
  python predict_nlu.py --interactive --device cuda
"""

import os
import json
import argparse

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# ─────────────────────────────────────────────
# 1. 加载模型
# ─────────────────────────────────────────────
def load_model(model_dir, device):
    """
    从 model_dir 目录加载训练好的 BERT 分类模型。

    model_dir : 训练脚本保存模型的目录（默认 ./nlu_model）
    device    : torch.device 对象，指定在 CPU 还是 GPU 上运行

    返回：
        model     - 加载好的分类模型
        tokenizer - 分词器（把中文句子切成 token）
        id2label  - 整数 id → 意图分类名 的映射字典
    """
    print(f"  加载 tokenizer ...")
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    print(f"  加载模型权重 ...")
    model = BertForSequenceClassification.from_pretrained(model_dir)

    # 把模型移动到指定设备（CPU 或 GPU）
    # 类比 Android：把任务分配到主线程还是子线程
    model.to(device)

    # 切换到推理模式：关闭 Dropout，不计算梯度（更快、省内存）
    # 类比：Android 的 Release Build，去掉调试开销
    model.eval()

    # 加载标签映射文件（训练结束时保存的 label_mapping.json）
    label_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # json 的 key 是字符串，需要转回整数
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        # 兜底：从模型自带的 config 里读标签（如果 label_mapping.json 丢失）
        id2label = {i: model.config.id2label[i] for i in range(model.config.num_labels)}

    return model, tokenizer, id2label


# ─────────────────────────────────────────────
# 2. 单条预测
# ─────────────────────────────────────────────
def predict(text, model, tokenizer, id2label, device, max_len=64):
    """
    对单条话语进行意图分类。

    text      : 用户输入的话语，例如 "打开空调"
    model     : 加载好的 BERT 模型
    tokenizer : 分词器
    id2label  : id → 分类名 字典
    device    : 运行设备
    max_len   : 最大 token 长度（和训练时保持一致）

    返回：
        pred_label  - 预测的意图分类名，例如 "直接车控"
        confidence  - 预测置信度（0~1），例如 0.98
        all_probs   - 所有分类的概率字典，例如 {"直接车控": 0.98, "闲聊": 0.01, ...}
    """
    # 分词：把中文句子转成模型需要的 input_ids、attention_mask 等
    inputs = tokenizer(
        text,
        truncation=True,        # 超长截断
        max_length=max_len,
        return_tensors="pt",    # 返回 PyTorch tensor
    )

    # 把输入数据也移动到同一设备（和模型保持一致，否则会报错）
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # torch.no_grad()：推理时不需要计算梯度，节省内存和时间
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits              # 原始评分（未归一化）
        # softmax 把评分转成概率分布，所有类别概率之和为 1
        probs = torch.softmax(logits, dim=-1).squeeze()

    # 取概率最高的类别作为预测结果
    pred_id = int(logits.argmax(dim=-1).item())
    pred_label = id2label[pred_id]
    confidence = float(probs[pred_id])

    # 整理所有类别的概率，方便展示 Top-K
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(id2label))}

    return pred_label, confidence, all_probs


# ─────────────────────────────────────────────
# 3. 格式化打印预测结果
# ─────────────────────────────────────────────
def print_result(text, pred_label, confidence, all_probs, topk=3):
    """打印预测结果和 Top-K 候选"""
    print(f"\n  话语：{text}")
    print(f"  预测：{pred_label}  （置信度：{confidence:.1%}）")

    # 按概率从高到低排序，取前 topk 个
    sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
    print(f"  Top-{topk} 候选：")
    for rank, (label, prob) in enumerate(sorted_probs[:topk], 1):
        # 用方块字符画一个简单的进度条，直观展示概率大小
        bar = "█" * int(prob * 25)
        print(f"    {rank}. {label:12s}  {prob:5.1%}  {bar}")


# ─────────────────────────────────────────────
# 4. 主函数
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NLU 意图分类预测")
    parser.add_argument(
        "--model_dir", type=str, default="./nlu_model",
        help="训练好的模型目录（默认：./nlu_model）"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="运行设备：auto=自动检测 / cpu / cuda（默认：auto）"
    )
    parser.add_argument(
        "--text", type=str, action="append",
        help="要预测的话语（可以多次使用，例如：--text '打开空调' --text '导航'）"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="交互模式：循环输入，适合批量测试"
    )
    parser.add_argument(
        "--topk", type=int, default=3,
        help="显示概率最高的前 K 个候选（默认：3）"
    )
    parser.add_argument(
        "--max_len", type=int, default=64,
        help="最大 token 长度，需和训练时一致（默认：64）"
    )
    args = parser.parse_args()

    # ── 确定运行设备 ────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"设备检测：使用 {device}")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("⚠ 指定了 cuda 但未检测到 GPU，回退到 CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        print(f"运行设备：{device}")

    # ── 检查模型目录是否存在 ────────────────────
    if not os.path.isdir(args.model_dir):
        print(f"\n❌ 模型目录不存在：{args.model_dir}")
        print("   请先运行 train_nlu.py 完成训练，再使用本脚本预测。")
        return

    # ── 加载模型 ────────────────────────────────
    print(f"\n加载模型：{args.model_dir}")
    model, tokenizer, id2label = load_model(args.model_dir, device)
    print(f"  共 {len(id2label)} 个意图分类：{list(id2label.values())}")

    # ── 交互模式 ────────────────────────────────
    if args.interactive:
        print("\n" + "=" * 55)
        print("  NLU 意图分类 · 交互模式")
        print("  输入话语后按 Enter，输入 'q' 或 'quit' 退出")
        print("=" * 55)
        while True:
            try:
                text = input("\n话语 > ").strip()
            except (EOFError, KeyboardInterrupt):
                # 处理 Ctrl+C 或管道结束，优雅退出
                print("\n已退出。")
                break

            if not text:
                continue
            if text.lower() in ["q", "quit", "exit", "退出"]:
                print("已退出。")
                break

            pred_label, confidence, all_probs = predict(
                text, model, tokenizer, id2label, device, args.max_len
            )
            print_result(text, pred_label, confidence, all_probs, args.topk)

    # ── 命令行批量模式 ───────────────────────────
    elif args.text:
        print()
        for text in args.text:
            pred_label, confidence, all_probs = predict(
                text, model, tokenizer, id2label, device, args.max_len
            )
            print_result(text, pred_label, confidence, all_probs, args.topk)

    # ── 无参数：打印帮助 ─────────────────────────
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
