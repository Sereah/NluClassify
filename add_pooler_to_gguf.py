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
  2. 已执行 convert_hf_to_gguf.py，生成了 nlu_model.gguf
  3. llama.cpp 已克隆到 ./llama.cpp/

用法：
  python add_pooler_to_gguf.py

输入：nlu_model-bert-base-chinese-F32.gguf + nlu_model/model.safetensors
输出：nlu_model-bert-base-chinese-F32-pooler.gguf
"""

import sys
import numpy as np

# 把 llama.cpp 里的 gguf-py 加到 Python 模块搜索路径
# 类比 Android：这相当于在 build.gradle 里 implementation 一个本地 module
sys.path.insert(0, "./llama.cpp/gguf-py")

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from safetensors.torch import load_file

# ── 文件路径配置 ──────────────────────────────
INPUT_GGUF  = "nlu_model-bert-base-chinese-F32.gguf"         # convert_hf_to_gguf.py 生成的原始 GGUF
OUTPUT_GGUF = "nlu_model-bert-base-chinese-F32-pooler.gguf"  # 追加 pooler 权重后的新 GGUF
HF_MODEL    = "nlu_model/model.safetensors"                   # 训练后 HuggingFace 格式的权重文件

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

# ── 追加 pooler 权重作为 KV 元数据 ───────────
# 选择用 KV 方式存储（而非新增张量），是因为 llama.cpp 推理时不会加载多余张量
print("追加 pooler 权重到 KV 元数据...")
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
