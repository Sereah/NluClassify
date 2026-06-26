"""
test_gguf.py
============
作用：加载含 pooler 层的 GGUF 文件，对用户输入的话语做意图分类推理。
     完全基于 GGUF 文件，不依赖任何 HuggingFace 模型文件。

推理原理：
  1. 用 llama-cpp-python 加载 GGUF，提取 [CLS] token 的向量（768 维）
  2. 手动还原 pooler 层：pooler_out = tanh(W @ cls_vec + b)
  3. 手动还原分类头：logits = cls_w @ pooler_out + cls_b
  4. softmax 得到各意图概率

执行前提：
  1. 已完成 add_pooler_to_gguf.py，生成了 *-pooler.gguf 文件
  2. 已安装 llama-cpp-python：pip install llama-cpp-python

用法：
  # 默认加载 F32 版本
  python test_gguf.py

  # 指定量化版本
  python test_gguf.py --model nlu_model-bert-base-chinese-Q8_0-pooler.gguf
"""

import sys
import argparse
import numpy as np

# 把 llama.cpp 里的 gguf-py 加到 Python 模块搜索路径
sys.path.insert(0, "./llama.cpp/gguf-py")

from llama_cpp import Llama
from gguf import GGUFReader

# ── 命令行参数 ────────────────────────────────
parser = argparse.ArgumentParser(description="BERT GGUF 意图分类推理")
parser.add_argument(
    "--model", "-m",
    default="nlu_model-bert-base-chinese-F32-pooler.gguf",
    help="GGUF 模型文件路径（默认：nlu_model-bert-base-chinese-F32-pooler.gguf）",
)
args = parser.parse_args()

# ── 配置 ─────────────────────────────────────
GGUF_PATH = args.model   # 包含 pooler 权重的 GGUF 文件
TOP_K     = 3            # 显示概率最高的前 K 个意图

# ── 读取 GGUF 里的元数据 ──────────────────────
print(f"读取 GGUF 文件：{GGUF_PATH}")
r = GGUFReader(GGUF_PATH)
fields  = r.fields
weights = {t.name: np.array(t.data) for t in r.tensors}

# 从 GGUF 的 KV 元数据里读取标签列表
# convert_hf_to_gguf.py 会把模型 config.json 里的 id2label 写进 GGUF
id2label = {
    i: s for i, s in enumerate(fields["bert.classifier.output_labels"].contents())
}
num_labels = len(id2label)
print(f"意图分类数量：{num_labels}")
print(f"分类列表：{list(id2label.values())}")

# 从 KV 元数据里还原 pooler 层权重
# pooler_w 形状：[768, 768]，pooler_b 形状：[768]
pooler_w = np.array(
    fields["bert.classifier.pooler_weight"].contents(), dtype=np.float32
).reshape(768, 768)
pooler_b = np.array(
    fields["bert.classifier.pooler_bias"].contents(), dtype=np.float32
)

# 从张量里读取分类头权重
# cls.output.weight 形状：[num_labels, 768]
# cls.output.bias   形状：[num_labels]
cls_w = weights["cls.output.weight"]   # [num_labels, 768]
cls_b = weights["cls.output.bias"]     # [num_labels]
print(f"分类头权重形状：{cls_w.shape}（{num_labels} 个类 × 768 维）")

# ── 加载 llama-cpp-python 推理引擎 ────────────
# embedding=True：让模型输出向量而不是生成文字
# pooling_type=2：使用 [CLS] token 的向量作为句子表示（BERT 的标准做法）
# n_ctx=512：最大上下文长度（128 就够，设 512 留余量）
print("\n加载推理引擎...")
model = Llama(
    model_path=GGUF_PATH,
    embedding=True,
    pooling_type=2,
    n_ctx=512,
    verbose=False,      # 关掉 llama.cpp 的详细日志，保持输出整洁
)
print("加载完成！\n")


def predict(text):
    """
    对一条话语做意图分类，返回 (预测分类名, 置信度, 所有分类概率字典)

    内部流程：
      text → tokenize → BERT encoder → [CLS] 向量
           → pooler 层（tanh 激活）
           → 分类头（线性层）
           → softmax → 概率分布
    """
    # 第一步：用 BERT 提取句子向量（[CLS] token 的输出，768 维）
    cls_hidden = np.array(model.embed(text), dtype=np.float32)

    # 第二步：pooler 层变换（还原 bert.pooler.dense + tanh）
    # 类比：这是 BERT 顶部的一个全连接层，专门对 [CLS] 向量做语义压缩
    pooler_out = np.tanh(pooler_w @ cls_hidden + pooler_b)

    # 第三步：分类头（线性层，无激活）
    logits = cls_w @ pooler_out + cls_b

    # 第四步：softmax 得到概率分布
    # 先减去最大值是为了数值稳定（防止 exp 溢出，结果不变）
    exp   = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    # 整理成字典，按概率排序
    all_probs = {id2label[i]: float(probs[i]) for i in range(num_labels)}
    ranked    = sorted(all_probs.items(), key=lambda x: -x[1])

    pred_label  = ranked[0][0]
    confidence  = ranked[0][1]
    return pred_label, confidence, all_probs


def print_result(text, pred_label, confidence, all_probs):
    """格式化打印预测结果"""
    print(f"\n  话语：{text}")
    print(f"  预测：{pred_label}  （置信度：{confidence:.1%}）")
    ranked = sorted(all_probs.items(), key=lambda x: -x[1])
    print(f"  Top-{TOP_K} 候选：")
    for rank, (label, prob) in enumerate(ranked[:TOP_K], 1):
        bar = "█" * int(prob * 25)
        print(f"    {rank}. {label:12s}  {prob:5.1%}  {bar}")


# ── 交互推理循环 ──────────────────────────────
print("=" * 50)
print("  NLU GGUF 推理  （输入 q 退出）")
print("=" * 50)

while True:
    try:
        text = input("\n话语 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        break

    if not text:
        continue
    if text.lower() in ("q", "quit", "exit", "退出"):
        print("已退出。")
        break

    pred_label, confidence, all_probs = predict(text)
    print_result(text, pred_label, confidence, all_probs)
