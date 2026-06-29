#!/usr/bin/env bash
# setup_env.sh — 检查并安装 nlu_classify 所需依赖
# 用法：bash setup_env.sh

set -euo pipefail

# ── 颜色 ────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# ── 需要的精确版本 ───────────────────────────────────────
declare -A REQUIRED_VERSIONS=(
    [torch]="2.3.0"
    [transformers]="4.47.0"
    [numpy]="1.26.4"
    [pandas]="2.2.2"
    [scikit-learn]="1.5.1"
    [accelerate]="0.33.0"
    [tokenizers]="0.19.1"
)

# pip install 时部分包名与 import 名不同，此处统一用 pip 包名
# scikit-learn 在 pip 里叫 scikit-learn，show 时也叫 scikit-learn，无需额外处理

echo ""
echo "============================================================"
echo "   NLU Classify — 环境检查"
echo "============================================================"
echo ""

# ── Step 1：检查是否在 conda 环境中 ────────────────────
echo "[1/3] 检查 conda 环境"

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo -e "${RED}[错误] 未检测到激活的 conda 环境。${NC}"
    echo ""
    echo "请先激活一个 conda 环境后再运行此脚本，例如："
    echo ""
    echo "  conda activate voice"
    echo "  bash setup_env.sh"
    echo ""
    echo "如果还没有合适的环境，可以先创建一个："
    echo ""
    echo "  conda create -n voice python=3.10"
    echo "  conda activate voice"
    echo "  bash setup_env.sh"
    echo ""
    exit 1
fi

if [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
    echo -e "${YELLOW}[警告] 当前在 conda base 环境中。${NC}"
    echo "       建议使用独立环境，避免污染 base 依赖。"
    echo ""
    printf "       是否仍在 base 环境中继续安装？[y/N] "
    read -r answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo ""
        echo "已取消。请激活其他 conda 环境后重试。"
        exit 0
    fi
    echo ""
fi

echo -e "  当前 conda 环境：${GREEN}${CONDA_DEFAULT_ENV}${NC}"
echo ""

# ── Step 2：检查 Python / pip ───────────────────────────
echo "[2/3] 检查 Python 与 pip"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[错误] 未找到 python3，请检查 conda 环境是否正确安装。${NC}"
    exit 1
fi

if ! command -v pip &>/dev/null; then
    echo -e "${RED}[错误] 未找到 pip，请检查 conda 环境是否正确安装。${NC}"
    exit 1
fi

PYTHON_VER=$(python3 --version 2>&1)
echo "  $PYTHON_VER"
echo ""

# ── Step 3：逐包检查版本 ────────────────────────────────
echo "[3/3] 检查依赖包版本"
echo ""
printf "  %-16s %-14s %-14s %s\n" "包名" "需要版本" "已安装版本" "状态"
printf "  %-16s %-14s %-14s %s\n" "----------------" "--------------" "--------------" "------"

MISSING_PKGS=()
WRONG_PKGS=()

version_ge() {
    python3 -c "
a = tuple(int(x) for x in '$1'.split('.')[:3])
b = tuple(int(x) for x in '$2'.split('.')[:3])
exit(0 if a >= b else 1)
" 2>/dev/null
}

version_eq() {
    python3 -c "
a = tuple(int(x) for x in '$1'.split('.')[:3])
b = tuple(int(x) for x in '$2'.split('.')[:3])
exit(0 if a == b else 1)
" 2>/dev/null
}

for pkg in torch transformers numpy pandas scikit-learn accelerate tokenizers; do
    required="${REQUIRED_VERSIONS[$pkg]}"
    installed=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')

    if [ -z "$installed" ]; then
        printf "  %-16s %-14s %-14s " "$pkg" "$required" "未安装"
        echo -e "${RED}✗ 缺失${NC}"
        MISSING_PKGS+=("$pkg==$required")
    elif version_eq "$installed" "$required"; then
        printf "  %-16s %-14s %-14s " "$pkg" "$required" "$installed"
        echo -e "${GREEN}✓${NC}"
    elif version_ge "$installed" "$required"; then
        printf "  %-16s %-14s %-14s " "$pkg" "$required" "$installed"
        echo -e "${YELLOW}⚠ 版本偏高${NC}"
        WRONG_PKGS+=("$pkg==$required")
    else
        printf "  %-16s %-14s %-14s " "$pkg" "$required" "$installed"
        echo -e "${RED}✗ 版本不足${NC}"
        WRONG_PKGS+=("$pkg==$required")
    fi
done

echo ""

# ── 汇总并询问用户 ──────────────────────────────────────
ALL_ISSUES=("${MISSING_PKGS[@]}" "${WRONG_PKGS[@]}")

if [ ${#ALL_ISSUES[@]} -eq 0 ]; then
    echo -e "${GREEN}所有依赖版本均符合要求，可直接运行 train_nlu.py。${NC}"
    echo ""
    exit 0
fi

# 缺失包
if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${RED}缺失的包：${NC}"
    for pkg in "${MISSING_PKGS[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    printf "是否安装以上缺失的包？[y/N] "
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo ""
        pip install "${MISSING_PKGS[@]}"
        echo ""
    else
        echo "已跳过缺失包的安装。"
        echo ""
    fi
fi

# 版本不符的包
if [ ${#WRONG_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}版本不符的包（将覆盖安装指定版本）：${NC}"
    for pkg in "${WRONG_PKGS[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    printf "是否覆盖安装以上包到指定版本？[y/N] "
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo ""
        pip install --force-reinstall "${WRONG_PKGS[@]}"
        echo ""
    else
        echo "已跳过版本覆盖安装。"
        echo ""
    fi
fi

echo -e "${GREEN}环境检查完成，可运行 train_nlu.py。${NC}"
echo ""
