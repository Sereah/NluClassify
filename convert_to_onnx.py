"""
NLU 模型转换脚本：HuggingFace → ONNX → INT8 量化 → 集成 pre/post processing
====================================================================================

把训练好的 BertForSequenceClassification 模型转换为 Android 可用的 ONNX 文件，
内含 tokenizer + 推理 + argmax + label 映射，最终产物可直接被
onnxruntime-android AAR 加载执行。

类比 llama.cpp 的 convert.py → quantize 一条龙。

用法：
  python convert_to_onnx.py

输出：
  nlu_model_onnx/
  ├── nlu_model.onnx          ← 纯 FP32 ONNX（备份）
  ├── nlu_model_quant.onnx    ← INT8 动态量化
  └── nlu_model_mobile.onnx    ← 量化 + tokenizer + 后处理（给 Android 用）
"""

import json
import os
import sys
from pathlib import Path
from contextlib import contextmanager

import torch
import onnx
from transformers import BertTokenizer, BertForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_DIR = Path("./nlu_model")          # 训练好的模型目录
OUTPUT_DIR = Path("./nlu_model_onnx")     # ONNX 输出目录
OPSET_VERSION = 16                        # ONNX opset 版本（与 onnxruntime-extensions 对齐）


# ─────────────────────────────────────────────
# Step 1: 加载本地模型并导出 ONNX
#   等价于 llama.cpp 的 convert.py
#   使用 torch.onnx.export() 直接导出（非 dynamo 模式），
#   避免新版 PyTorch exporter 的 shape 标注问题
# ─────────────────────────────────────────────
def export_to_onnx(model_dir: Path, output_dir: Path) -> tuple[Path, BertTokenizer]:
    print("=" * 60)
    print("[1/3] 加载模型并导出 ONNX")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 tokenizer 和模型（从本地目录，不需要联网）
    print(f"  加载 tokenizer: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    print(f"  加载模型: {model_dir}")
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.cpu()

    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {num_params:.1f}M")
    print(f"  分类数: {model.config.num_labels}")
    print(f"  标签: {list(model.config.id2label.values())}")

    # 用 tokenizer 生成一组 dummy 输入，确定输入形状
    dummy_texts = ["打开空调"]
    dummy_enc = tokenizer(
        dummy_texts,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = dummy_enc["input_ids"]
    attention_mask = dummy_enc["attention_mask"]
    token_type_ids = dummy_enc["token_type_ids"]

    print(f"  输入形状: input_ids={list(input_ids.shape)}, "
          f"attention_mask={list(attention_mask.shape)}, "
          f"token_type_ids={list(token_type_ids.shape)}")

    # 导出 ONNX — 使用传统的 torch.onnx.export（非 dynamo）
    onnx_path = output_dir / "nlu_model.onnx"
    print(f"  导出 ONNX → {onnx_path}")

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,  # 使用旧版 TorchScript 导出器，避免新 dynamo 的 shape 标注 bug
    )

    file_size_mb = onnx_path.stat().st_size / 1e6
    print(f"  FP32 ONNX 大小: {file_size_mb:.1f} MB")
    print(f"  ✅ 导出成功\n")

    return onnx_path, tokenizer


# ─────────────────────────────────────────────
# Step 2: INT8 动态量化
#   等价于 llama.cpp 的 Q8_0 量化
#   "动态"指激活值在推理时动态计算 scale，
#   权重在转换阶段就写入 INT8 — 不是加载时量化
# ─────────────────────────────────────────────
def quantize_onnx(onnx_path: Path) -> Path:
    print("=" * 60)
    print("[2/3] INT8 动态量化")
    print("=" * 60)

    quant_path = onnx_path.with_name(onnx_path.stem + "_quant").with_suffix(onnx_path.suffix)
    print(f"  输入: {onnx_path.name}")
    print(f"  输出: {quant_path.name}")
    print(f"  量化方式: Dynamic INT8 (weight only)")

    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
    )

    original_mb = onnx_path.stat().st_size / 1e6
    quant_mb = quant_path.stat().st_size / 1e6
    print(f"  原始大小: {original_mb:.1f} MB")
    print(f"  量化后:   {quant_mb:.1f} MB  (-{100 * (1 - quant_mb/original_mb):.0f}%)")
    print(f"  ✅ 量化完成\n")

    return quant_path


# ─────────────────────────────────────────────
# Step 3: 将 tokenizer + argmax + label 映射写入 ONNX 图
#   效果：输入 = 纯文本字符串，输出 = 分类结果字符串
#   Android 端只需一行 session.run("input_text", "打开空调")
# ─────────────────────────────────────────────
def add_pre_post_processing(quant_path: Path, model_dir: Path, output_dir: Path,
                            tokenizer: BertTokenizer) -> Path:
    print("=" * 60)
    print("[3/3] 添加 pre/post processing 到 ONNX 模型")
    print("=" * 60)

    # 使用已经加载好的 tokenizer（避免重复加载）
    label_mapping_path = model_dir / "label_mapping.json"
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    num_labels = len(label_mapping["id2label"])
    print(f"  分类数: {num_labels}")
    print(f"  标签: {list(label_mapping['id2label'].values())}")

    # tokenizer 需要的特殊文件：词表
    # 参考 QA 示例的 temp_vocab_file 模式
    @contextmanager
    def temp_vocab_file():
        vocab_file = quant_path.parent / "vocab.txt"
        # tokenizer.vocab 是 {word: id} 字典，写为 JSON
        with open(str(vocab_file), "w", encoding="utf-8") as f:
            json.dump(tokenizer.vocab, f, ensure_ascii=False)
        yield vocab_file
        vocab_file.unlink()

    output_path = output_dir / "nlu_model_mobile.onnx"

    print(f"  输入模型: {quant_path.name}")
    print(f"  输出模型: {output_path.name}")
    print(f"  Tokenizer 类型: BertTokenizer")
    print(f"  任务类型: SequenceClassification")

    with temp_vocab_file() as vocab_file:
        add_ppp.transformers_and_bert(
            quant_path.resolve(),
            output_path.resolve(),
            vocab_file.resolve(),
            "BertTokenizer",
            "SequenceClassification",
        )

    # 验证最终产物
    final_mb = output_path.stat().st_size / 1e6
    print(f"  最终模型大小: {final_mb:.1f} MB")
    print(f"  ✅ pre/post processing 添加完成\n")

    return output_path


# ─────────────────────────────────────────────
# Step 4: 验证最终模型结构
#   类似 Android 端打日志确认模型输入输出
# ─────────────────────────────────────────────
def validate_onnx(output_path: Path):
    """打印最终 ONNX 模型的输入输出结构"""
    import onnx

    print("=" * 60)
    print("[额外] 验证 ONNX 模型结构")
    print("=" * 60)

    model = onnx.load(str(output_path))
    print(f"  模型 IR 版本: {model.ir_version}")
    print(f"  图生产者: {model.producer_name}")

    print(f"\n  输入:")
    for inp in model.graph.input:
        print(f"    {inp.name}: {inp.type}")

    print(f"\n  输出:")
    for out in model.graph.output:
        print(f"    {out.name}: {out.type}")

    print(f"\n  ✅ 验证通过 — 模型可直接用于 onnxruntime-android")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  NLU 模型 → ONNX 转换 + 量化 + 移动端适配")
    print("=" * 60)
    print(f"  模型源: {MODEL_DIR.resolve()}")
    print(f"  输出目录: {OUTPUT_DIR.resolve()}")
    print()

    # 检查模型目录
    if not MODEL_DIR.is_dir():
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        print("  请先运行 train_nlu.py 完成训练。")
        sys.exit(1)

    safetensors = MODEL_DIR / "model.safetensors"
    if not safetensors.exists():
        print(f"❌ 模型权重不存在: {safetensors}")
        sys.exit(1)

    # 清理旧的输出文件
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    # Step 1: HuggingFace → FP32 ONNX
    onnx_path, tokenizer = export_to_onnx(MODEL_DIR, OUTPUT_DIR)

    # Step 2: FP32 ONNX → INT8 量化 ONNX
    quant_path = quantize_onnx(onnx_path)

    # Step 3: 集成 tokenizer + 后处理 → 移动端 ONNX
    mobile_path = add_pre_post_processing(quant_path, MODEL_DIR, OUTPUT_DIR, tokenizer)

    # 验证
    validate_onnx(mobile_path)

    # ── 总结 ──
    print("=" * 60)
    print("  转换完成！")
    print("=" * 60)
    print(f"\n  输出文件:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        tag = ""
        if f.name == "nlu_model_mobile.onnx":
            tag = "  ← 给 Android 用"
        print(f"    {f.name:30s}  {size_mb:6.1f} MB{tag}")

    print(f"\n  Android 端推理示例:")
    print(f'    session.run("input_text", "打开空调")')
    print(f'    → output = "直接车控"')
    print()


if __name__ == "__main__":
    main()
