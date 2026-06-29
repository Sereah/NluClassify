"""
patch_llama_cpp.py
==================
作用：修改 llama.cpp/conversion/base.py，让 convert_hf_to_gguf.py
     能够识别 bert-base-chinese 的分词器，不再报错。

问题原因：
  convert_hf_to_gguf.py 通过对测试字符串编码后求 SHA256 哈希来
  识别不同模型的分词器类型。bert-base-chinese 的哈希值尚未被收录，
  导致抛出 NotImplementedError，并在警告信息中打印出实际哈希值。

修复方案：
  直接运行 convert_hf_to_gguf.py（允许失败），从其警告输出里捕获
  "chkhsh: <hash>" 这一行，再将该哈希写入 base.py。
  这样无论 llama.cpp 版本怎么更新、chktxt 如何定义，都能拿到正确哈希。

用法：
  python patch_llama_cpp.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

# ── 路径配置 ──────────────────────────────────
TARGET_FILE    = "llama.cpp/conversion/base.py"
CONVERT_SCRIPT = "llama.cpp/convert_hf_to_gguf.py"
MODEL_DIR      = "./nlu_model"
FALLBACK_DIR   = "./bert-base-chinese"
INSERTION_MARKER = "        if res is None:"


def get_hash_from_conversion(model_dir: str) -> str | None:
    """
    运行转换脚本，从警告信息里提取 chkhsh。
    转换会因为 hash 未识别而报错——正好从 stderr 里捕获这个 hash。
    如果转换直接成功（hash 已被识别），返回 None 表示无需 patch。
    """
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_out = f.name

    try:
        result = subprocess.run(
            [sys.executable, CONVERT_SCRIPT, model_dir,
             "--outtype", "f32", "--outfile", tmp_out],
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stdout + result.stderr
        m = re.search(r"chkhsh:\s+([a-f0-9]{64})", combined)
        if m:
            return m.group(1)
        # 转换成功（无 chkhsh 报错）→ 已经 patch 过了
        return None
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


# ── 检查必要文件 ──────────────────────────────
if not os.path.exists(TARGET_FILE):
    print(f"❌ 找不到文件：{TARGET_FILE}")
    print("   请确认 llama.cpp 已克隆到 ./llama.cpp/")
    sys.exit(1)

if not os.path.exists(CONVERT_SCRIPT):
    print(f"❌ 找不到转换脚本：{CONVERT_SCRIPT}")
    sys.exit(1)

# 优先用训练后的模型，回退到原始预训练模型
if os.path.exists(MODEL_DIR):
    probe_dir = MODEL_DIR
elif os.path.exists(FALLBACK_DIR):
    probe_dir = FALLBACK_DIR
    print(f"  nlu_model 不存在，使用 {FALLBACK_DIR} 作为探测模型")
else:
    print("❌ 找不到 nlu_model 或 bert-base-chinese，请先训练模型或克隆预训练权重")
    sys.exit(1)

# ── 读取 base.py ──────────────────────────────
with open(TARGET_FILE, "r", encoding="utf-8") as f:
    content = f.read()

if INSERTION_MARKER not in content:
    print(f"❌ 未找到插入位置：'{INSERTION_MARKER}'")
    print("   llama.cpp 版本可能不兼容，请手动检查 base.py")
    sys.exit(1)

# ── 运行转换脚本捕获 hash ──────────────────────
print(f"运行转换脚本探测哈希（使用 {probe_dir}）...")
chkhsh = get_hash_from_conversion(probe_dir)

if chkhsh is None:
    print("✅ 转换脚本运行正常，base.py 无需 patch。")
    sys.exit(0)

print(f"  捕获到哈希：{chkhsh}")

# ── 检查是否已经 patch 过 ─────────────────────
if chkhsh in content:
    print("✅ 该哈希已在 base.py 中，无需重复 patch。")
else:
    backup_path = TARGET_FILE + ".bak"
    shutil.copy2(TARGET_FILE, backup_path)
    print(f"  已备份原文件：{backup_path}")

    new_entry = (
        f'        if chkhsh == "{chkhsh}":\n'
        f'            # ref: https://huggingface.co/google-bert/bert-base-chinese\n'
        f'            res = "bert-bge"\n'
    )
    content = content.replace(INSERTION_MARKER, new_entry + INSERTION_MARKER, 1)

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ patch 完成：{chkhsh} → {TARGET_FILE}")

print("\n现在可以执行转换命令：")
print("  python llama.cpp/convert_hf_to_gguf.py \\")
print("      nlu_model --outtype q8_0 --outfile nlu_model-bert-base-chinese-Q8_0.gguf")
