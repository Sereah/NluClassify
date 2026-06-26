"""
patch_llama_cpp.py
==================
作用：修改 llama.cpp/conversion/base.py，让 convert_hf_to_gguf.py
     能够识别 bert-base-chinese 的分词器，不再报错。

问题原因：
  convert_hf_to_gguf.py 通过对测试字符串编码后求 SHA256 哈希来
  识别不同模型的分词器类型。bert-base-chinese 的哈希值尚未被收录，
  导致抛出 NotImplementedError。

修复方案：
  运行时从本地 bert-base-chinese/ 加载 tokenizer，
  按照 base.py 完全相同的方式动态计算哈希值，再将其写入 base.py。
  这样无论 llama.cpp 版本怎么更新，哈希始终和当前版本匹配。

用法：
  python patch_llama_cpp.py
"""

import os
import re
import shutil
from hashlib import sha256

# ── 路径配置 ──────────────────────────────────
TARGET_FILE  = "llama.cpp/conversion/base.py"
BERT_MODEL   = "./bert-base-chinese"
# 找到插入位置的标志字符串（把新条目插在这一行之前）
INSERTION_MARKER = "        if res is None:"


def extract_chktxt(base_py_content: str) -> str:
    """
    从 base.py 源码里提取 chktxt 变量的值。
    这样即使 llama.cpp 升级修改了测试字符串，这里也能跟着同步，
    不需要手动更新 patch 脚本。
    """
    # 匹配 chktxt = '...' 或 chktxt = "..."（单行赋值）
    m = re.search(r"chktxt\s*=\s*(['\"])(.*?)\1", base_py_content, re.DOTALL)
    if m:
        return m.group(2)

    # 如果是多行字符串或其他格式，退出并提示手动处理
    raise RuntimeError(
        "无法从 base.py 提取 chktxt 变量，llama.cpp 的格式可能发生了变化，"
        "请手动检查 base.py 中的 get_vocab_base_pre() 函数。"
    )


def compute_hash(chktxt: str, tokenizer) -> str:
    """
    用本地 tokenizer 对 chktxt 编码后求 SHA256，
    完全复现 base.py 里 get_vocab_base_pre() 的计算过程。
    """
    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()
    return chkhsh


# ── 检查文件是否存在 ──────────────────────────
if not os.path.exists(TARGET_FILE):
    print(f"❌ 找不到文件：{TARGET_FILE}")
    print("   请确认 llama.cpp 已克隆到 ./llama.cpp/")
    exit(1)

if not os.path.exists(BERT_MODEL):
    print(f"❌ 找不到预训练模型：{BERT_MODEL}")
    print("   请先执行：git clone https://www.modelscope.cn/google-bert/bert-base-chinese.git")
    exit(1)

# ── 读取 base.py ──────────────────────────────
with open(TARGET_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ── 提取 chktxt ───────────────────────────────
print("从 base.py 提取测试字符串（chktxt）...")
try:
    chktxt = extract_chktxt(content)
    print(f"  提取成功，长度 {len(chktxt)} 字符")
except RuntimeError as e:
    print(f"❌ {e}")
    exit(1)

# ── 加载 tokenizer 并计算哈希 ─────────────────
print(f"加载 tokenizer：{BERT_MODEL}")
# 延迟导入，避免没装 transformers 时脚本一开始就崩溃
try:
    from transformers import AutoTokenizer
except ImportError:
    print("❌ 未找到 transformers 库，请先安装：pip install transformers")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
chkhsh = compute_hash(chktxt, tokenizer)
print(f"计算得到哈希：{chkhsh}")

# ── 检查是否已经 patch 过 ─────────────────────
if chkhsh in content:
    print("✅ 当前哈希已在 base.py 中，无需重复 patch。")
    exit(0)

# ── 检查插入位置 ──────────────────────────────
if INSERTION_MARKER not in content:
    print(f"❌ 未找到插入位置：'{INSERTION_MARKER}'")
    print("   llama.cpp 版本可能不兼容，请手动检查 base.py")
    exit(1)

# ── 备份并写入 ────────────────────────────────
backup_path = TARGET_FILE + ".bak"
shutil.copy2(TARGET_FILE, backup_path)
print(f"已备份原文件：{backup_path}")

new_entry = (
    f'        if chkhsh == "{chkhsh}":\n'
    f'            # ref: https://huggingface.co/google-bert/bert-base-chinese\n'
    f'            res = "bert-bge"\n'
)

new_content = content.replace(INSERTION_MARKER, new_entry + INSERTION_MARKER, 1)

with open(TARGET_FILE, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"✅ patch 完成：{TARGET_FILE}")
print(f"   哈希：{chkhsh}")
print(f"   分词器类型：bert-bge（WordPiece，与 bert-base-chinese 同类）")
print()
print("现在可以执行转换命令：")
print("  conda run -n <ENV> python llama.cpp/convert_hf_to_gguf.py \\")
print("      nlu_model --outtype f32 --outfile nlu_model-bert-base-chinese-F32.gguf")
