"""
add_pooler_to_gguf.py
=====================
作用：把 BERT 的 pooler 层权重从 safetensors 文件里读出来，
     追加写入已有的 GGUF 文件，生成新的 GGUF。

为什么需要这一步？
  llama.cpp 的 convert_hf_to_gguf.py 转换 BERT 时，只保留了主体权重，
  没有保留 pooler 层（bert.pooler.dense.weight / bias）。
  但推理分类时需要用 pooler 对 [CLS] 向量做变换，所以要手动补进去。

执行前提：
  1. 已完成训练，nlu_model/ 目录存在
  2. 已执行 convert_hf_to_gguf.py，生成了对应精度的 GGUF
  3. llama.cpp 已克隆到 ./llama.cpp/

用法：
  # 默认处理 F32 文件
  python add_pooler_to_gguf.py

  # 指定输入文件（量化版本）
  python add_pooler_to_gguf.py --input nlu_model-bert-base-chinese-Q8_0.gguf

  # 同时指定输入和输出
  python add_pooler_to_gguf.py \
      --input  nlu_model-bert-base-chinese-Q8_0.gguf \
      --output nlu_model-bert-base-chinese-Q8_0-pooler.gguf

输出文件名规则（不指定 --output 时）：
  xxx.gguf → xxx-pooler.gguf
"""

import sys
import argparse
import numpy as np

# 把 llama.cpp 里的 gguf-py 加到 Python 模块搜索路径
# 类比 Android：这相当于在 build.gradle 里 implementation 一个本地 module
sys.path.insert(0, "./llama.cpp/gguf-py")

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from safetensors.torch import load_file

# ── 命令行参数 ────────────────────────────────
parser = argparse.ArgumentParser(description="向 GGUF 文件追加 BERT pooler 层权重")
parser.add_argument(
    "--input", "-i",
    default="nlu_model-bert-base-chinese-F32.gguf",
    help="输入 GGUF 文件路径（默认：nlu_model-bert-base-chinese-F32.gguf）",
)
parser.add_argument(
    "--output", "-o",
    default=None,
    help="输出 GGUF 文件路径（默认：在输入文件名末尾加 -pooler，如 Q8_0.gguf → Q8_0-pooler.gguf）",
)
parser.add_argument(
    "--hf-model",
    default="nlu_model/model.safetensors",
    help="HuggingFace 格式权重文件路径（默认：nlu_model/model.safetensors）",
)
args = parser.parse_args()

# ── 文件路径配置 ──────────────────────────────
INPUT_GGUF = args.input
HF_MODEL   = args.hf_model

# 未指定输出时自动推导：在 .gguf 后缀前插入 -pooler
if args.output:
    OUTPUT_GGUF = args.output
elif INPUT_GGUF.endswith(".gguf"):
    OUTPUT_GGUF = INPUT_GGUF[:-5] + "-pooler.gguf"
else:
    OUTPUT_GGUF = INPUT_GGUF + "-pooler.gguf"

print(f"读取 HuggingFace 模型权重：{HF_MODEL}")
# load_file 读取 safetensors 格式的权重文件（比 .bin 格式更安全、更快）
hf = load_file(HF_MODEL)

# 提取 pooler 层的权重和偏置，转成 float32 列表
# pooler.dense.weight 形状是 [768, 768]（768 维 → 768 维的全连接层）
# pooler.dense.bias   形状是 [768]
pooler_w = hf["bert.pooler.dense.weight"].float().numpy().flatten().tolist()
pooler_b = hf["bert.pooler.dense.bias"].float().numpy().flatten().tolist()
print(f"pooler.weight: {len(pooler_w)} 个浮点数（应为 {768*768} = 768×768）")
print(f"pooler.bias:   {len(pooler_b)} 个浮点数（应为 768）")

# 提取分类头权重，转成 float32 列表
# classifier.weight 形状是 [num_labels, 768]
# classifier.bias   形状是 [num_labels]
# 注意：这里从 safetensors（F32）读取，而不是从 GGUF 张量读取。
# 原因：GGUF 张量会被量化（Q8_0/Q4_K_M），读回来的字节布局和 F32 不同，
# 直接当 float32 用会导致维度错误。存成 KV 条目始终保持 F32，
# 对所有量化精度（F32/F16/Q8_0/Q4_K_M）都能正确工作。
cls_w = hf["classifier.weight"].float().numpy().flatten().tolist()
cls_b = hf["classifier.bias"].float().numpy().flatten().tolist()
num_labels = hf["classifier.bias"].shape[0]
print(f"classifier.weight: {len(cls_w)} 个浮点数（{num_labels} 类 × 768 维）")
print(f"classifier.bias:   {len(cls_b)} 个浮点数（{num_labels} 类）")

print(f"\n读取原始 GGUF：{INPUT_GGUF}")
reader = GGUFReader(INPUT_GGUF)

# 获取模型架构名称（例如 "bert"），写新 GGUF 时需要
arch = reader.fields["general.architecture"].contents()
print(f"模型架构：{arch}")

print(f"创建新 GGUF：{OUTPUT_GGUF}")
writer = GGUFWriter(OUTPUT_GGUF, arch=arch, endianess=reader.endianess)

# ── 复制原有的所有 KV 元数据 ──────────────────
# KV 元数据包含：层数、注意力头数、词表大小、标签映射等配置信息
print("复制 KV 元数据...")
for field in reader.fields.values():
    # 跳过 architecture 字段（Writer 初始化时已自动写入）
    if field.name == "general.architecture" or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

# ── 追加 pooler 和分类头权重作为 KV 元数据 ────
# 全部用 KV（F32 数组）存储，不依赖 GGUF 张量。
# 原因：量化后（Q8_0/Q4_K_M）张量的字节布局不再是纯 float32，
# 读回来会得到错误的形状和数值。KV 条目始终存 F32，对所有精度都安全。
print("追加 pooler 和分类头权重到 KV 元数据...")
writer.add_key_value(
    "bert.classifier.pooler_weight",
    pooler_w,
    gguf.GGUFValueType.ARRAY,
    sub_type=gguf.GGUFValueType.FLOAT32,
)
writer.add_key_value(
    "bert.classifier.pooler_bias",
    pooler_b,
    gguf.GGUFValueType.ARRAY,
    sub_type=gguf.GGUFValueType.FLOAT32,
)
writer.add_key_value(
    "bert.classifier.output_weight",
    cls_w,
    gguf.GGUFValueType.ARRAY,
    sub_type=gguf.GGUFValueType.FLOAT32,
)
writer.add_key_value(
    "bert.classifier.output_bias",
    cls_b,
    gguf.GGUFValueType.ARRAY,
    sub_type=gguf.GGUFValueType.FLOAT32,
)

# ── 复制所有原有张量 ──────────────────────────
# 张量包含 BERT 各层的实际权重（embedding、attention、ffn 等）
print("复制张量数据...")
for tensor in reader.tensors:
    writer.add_tensor_info(
        tensor.name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        tensor.tensor_type,
    )

# 按顺序写文件（顺序不能乱，GGUF 格式要求 header → KV → tensor info → tensor data）
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()

for tensor in reader.tensors:
    writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

writer.close()
print(f"\n✅ 完成！输出文件：{OUTPUT_GGUF}")
