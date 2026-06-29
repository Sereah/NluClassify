#!/usr/bin/env bash
# setup_env.sh — 检查并安装 nlu_classify 所需依赖
# 用法：bash setup_env.sh

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# ── 兼容版本范围 [MIN, MAX) ────────────────────────────────
# 同一主版本内向下兼容，超出范围的包额外用探针实测关键 API
declare -A MIN_VER=(
    [torch]="2.3.0"
    [transformers]="4.47.0"
    [numpy]="1.24.0"
    [pandas]="1.5.0"
    [scikit-learn]="1.2.0"
    [accelerate]="0.20.0"
    [tokenizers]="0.13.0"
)

declare -A MAX_VER=(
    [torch]="4.0.0"
    [transformers]="6.0.0"
    [numpy]="3.0.0"
    [pandas]="4.0.0"
    [scikit-learn]="2.0.0"
    [accelerate]="2.0.0"
    [tokenizers]="1.0.0"
)

PKGS_ORDER=(torch transformers numpy pandas scikit-learn accelerate tokenizers)

# ── 版本比较 ───────────────────────────────────────────────
version_ge() {
    python3 -c "
a=tuple(int(x) for x in '$1'.split('.')[:3])
b=tuple(int(x) for x in '$2'.split('.')[:3])
exit(0 if a>=b else 1)" 2>/dev/null
}

version_lt() {
    python3 -c "
a=tuple(int(x) for x in '$1'.split('.')[:3])
b=tuple(int(x) for x in '$2'.split('.')[:3])
exit(0 if a<b else 1)" 2>/dev/null
}

get_installed() {
    pip show "$1" 2>/dev/null | grep "^Version:" | awk '{print $2}'
}

echo ""
echo "============================================================"
echo "   NLU Classify — 环境检查"
echo "============================================================"
echo ""

# ── Step 1：conda 环境 ─────────────────────────────────────
echo "[1/3] 检查 conda 环境"

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo -e "${RED}[错误] 未检测到激活的 conda 环境。${NC}"
    echo ""
    echo "请先激活 conda 环境后再运行此脚本，例如："
    echo ""
    echo "  conda activate <ENV>"
    echo "  bash setup_env.sh"
    echo ""
    echo "如还没有环境，可先创建："
    echo ""
    echo "  conda create -n <ENV> python=3.10"
    echo "  conda activate <ENV>"
    echo "  bash setup_env.sh"
    echo ""
    exit 1
fi

if [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
    echo -e "${YELLOW}[警告] 当前在 conda base 环境，建议使用独立环境避免污染。${NC}"
    echo ""
    printf "  是否仍在 base 环境中继续？[y/N] "
    read -r answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "已取消。"
        exit 0
    fi
    echo ""
fi

echo -e "  当前环境：${GREEN}${CONDA_DEFAULT_ENV}${NC}"
echo ""

# ── Step 2：Python / pip ───────────────────────────────────
echo "[2/3] 检查 Python 与 pip"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[错误] 未找到 python3。${NC}"
    exit 1
fi
if ! command -v pip &>/dev/null; then
    echo -e "${RED}[错误] 未找到 pip。${NC}"
    exit 1
fi

echo "  $(python3 --version)"
echo ""

# ── Step 3：版本范围检查 + 探针 ───────────────────────────
echo "[3/3] 检查依赖包版本"
echo ""
printf "  %-16s %-22s %-14s %s\n" "包名" "兼容范围" "已安装" "状态"
printf "  %-16s %-22s %-14s %s\n" "----------------" "----------------------" "--------------" "------"

NEED_INSTALL=()   # 缺失 / 版本过低
NEED_PROBE=()     # 版本超出范围上限，需探针实测

for pkg in "${PKGS_ORDER[@]}"; do
    min="${MIN_VER[$pkg]}"
    max="${MAX_VER[$pkg]}"
    range=">=${min}, <${max}"
    installed=$(get_installed "$pkg")

    if [ -z "$installed" ]; then
        printf "  %-16s %-22s %-14s " "$pkg" "$range" "未安装"
        echo -e "${RED}✗ 缺失${NC}"
        NEED_INSTALL+=("${pkg}>=${min},<${max}")
    elif version_lt "$installed" "$min"; then
        printf "  %-16s %-22s %-14s " "$pkg" "$range" "$installed"
        echo -e "${RED}✗ 版本过低${NC}"
        NEED_INSTALL+=("${pkg}>=${min},<${max}")
    elif version_ge "$installed" "$max"; then
        printf "  %-16s %-22s %-14s " "$pkg" "$range" "$installed"
        echo -e "${YELLOW}? 超出范围，探针验证中...${NC}"
        NEED_PROBE+=("$pkg")
    else
        printf "  %-16s %-22s %-14s " "$pkg" "$range" "$installed"
        echo -e "${GREEN}✓${NC}"
    fi
done

echo ""

# ── 探针：对超出版本上限的包实测关键 API ──────────────────
PROBE_FAILED=()

probe_pkg() {
    local pkg="$1"
    local code="$2"
    local label="$3"
    local result
    result=$(python3 -c "$code" 2>&1)
    if [[ "$result" == "OK" ]]; then
        echo -e "  ${GREEN}[✓]${NC} ${pkg}: ${label}"
    else
        echo -e "  ${RED}[✗]${NC} ${pkg}: ${result}"
        PROBE_FAILED+=("${pkg}>=${MIN_VER[$pkg]},<${MAX_VER[$pkg]}")
    fi
}

if [ ${#NEED_PROBE[@]} -gt 0 ]; then
    echo "  API 兼容性探针："
    for pkg in "${NEED_PROBE[@]}"; do
        case "$pkg" in
            transformers)
                probe_pkg "transformers" "
from transformers import Trainer
import inspect
params = inspect.signature(Trainer.__init__).parameters
if 'processing_class' not in params:
    print('Trainer 不支持 processing_class 参数（需要 transformers>=4.46.0）')
else:
    print('OK')
" "Trainer.processing_class 参数存在"
                ;;
            torch)
                probe_pkg "torch" "
import torch
_ = torch.tensor([1.0]) + torch.tensor([1.0])
print('OK')
" "基础张量运算正常"
                ;;
            numpy)
                probe_pkg "numpy" "
import numpy as np
_ = np.array([1.0, 2.0]).mean()
print('OK')
" "基础运算正常"
                ;;
            pandas)
                probe_pkg "pandas" "
import pandas as pd
_ = pd.DataFrame({'a': [1, 2]})
print('OK')
" "基础 DataFrame 正常"
                ;;
            scikit-learn)
                probe_pkg "scikit-learn" "
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
print('OK')
" "train_test_split / accuracy_score / f1_score 可用"
                ;;
            accelerate)
                probe_pkg "accelerate" "
import accelerate
print('OK')
" "导入正常"
                ;;
            tokenizers)
                probe_pkg "tokenizers" "
import tokenizers
print('OK')
" "导入正常"
                ;;
        esac
    done
    echo ""
fi

# ── 汇总处理 ──────────────────────────────────────────────
if [ ${#NEED_INSTALL[@]} -eq 0 ] && [ ${#PROBE_FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}所有依赖均符合要求，可直接运行 train_nlu.py。${NC}"
    echo ""
    exit 0
fi

if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
    echo -e "${RED}需要安装/升级的包：${NC}"
    for pkg in "${NEED_INSTALL[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    printf "是否安装以上包？[y/N] "
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo ""
        pip install "${NEED_INSTALL[@]}"
        echo ""
    else
        echo "已跳过。"
        echo ""
    fi
fi

if [ ${#PROBE_FAILED[@]} -gt 0 ]; then
    echo -e "${YELLOW}以下包版本超出兼容范围且探针验证失败，需覆盖安装：${NC}"
    for pkg in "${PROBE_FAILED[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    printf "是否覆盖安装到兼容版本？[y/N] "
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo ""
        pip install --force-reinstall "${PROBE_FAILED[@]}"
        echo ""
    else
        echo "已跳过。"
        echo ""
    fi
fi

echo -e "${GREEN}环境检查完成，可运行 train_nlu.py。${NC}"
echo ""
