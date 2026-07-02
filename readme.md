# NLU 意图分类 · 完整流程

基于 BERT 的车载语音意图分类，支持训练、HuggingFace 推理、导出 GGUF 嵌入式推理。

---

## 目录结构

```
nlu_classify/
├── setup_env.sh                               # 环境检查与自动安装脚本（首次使用先运行）
├── train_nlu.py                               # 训练脚本
├── predict_nlu.py                             # HuggingFace 格式推理
├── convert_to_onnx.py                         # HuggingFace → ONNX + 量化 + 移动端适配
├── add_pooler_to_gguf.py                      # 追加 pooler 层到 GGUF
├── test_gguf.py                               # GGUF 格式推理测试
├── patch_llama_cpp.py                         # 修复 llama.cpp 分词器识别问题
├── readme.md                                  # 本文档
├── training_data/                             # CSV 训练数据（每个文件一个意图分类）
│   ├── 感知车控.csv
│   ├── 直接车控.csv
│   ├── 车书.csv
│   ├── 影音.csv
│   ├── 出行.csv
│   ├── 闲聊.csv
│   └── 搜索.csv
├── bert-base-chinese/                         # 需手动克隆（见步骤零）
├── nlu_model/                                 # 训练后自动生成（HuggingFace 格式）
├── nlu_model_onnx/                            # ONNX 转换后生成（见路径 B）
│   ├── nlu_model.onnx                         #   FP32 原始导出（409 MB，备份）
│   ├── nlu_model_quant.onnx                   #   INT8 量化（103 MB）
│   └── nlu_model_mobile.onnx                  #   ★ 量化 + tokenizer + 后处理，Android 直接加载
├── llama.cpp/                                 # 需手动克隆（见路径 A）
├── nlu_model-bert-base-chinese-F32.gguf        # 转换后生成（按选择的量化精度命名）
└── nlu_model-bert-base-chinese-F32-pooler.gguf # 追加 pooler 后生成，用于最终推理
```

---

## 步骤零：克隆预训练模型

在 `nlu_classify/` 目录下执行：

```bash
git clone https://www.modelscope.cn/google-bert/bert-base-chinese.git
```

克隆完成后目录里会有 `bert-base-chinese/`，包含：
- `pytorch_model.bin` / `model.safetensors`：预训练权重
- `config.json`：模型配置
- `vocab.txt`：中文词表
- `tokenizer_config.json`：分词器配置

> 国内网络直接克隆 HuggingFace 较慢，ModelScope 镜像速度更快。

---

## 步骤一：准备 conda 环境

在 conda 环境中运行环境检查脚本，脚本会自动检测依赖版本，缺失或版本不符时提示你确认后安装：

```bash
conda activate <ENV>   # 替换为你自己的环境名，没有的话先 conda create -n <ENV> python=3.10
bash setup_env.sh
```

脚本会检查 `torch`、`transformers`、`numpy`、`pandas`、`scikit-learn`、`accelerate`、`tokenizers` 的版本，并在需要时引导你完成安装。

> **后续所有步骤均在同一 conda 环境中执行，不再重复写激活命令。**

---

## 步骤二：训练

在 `nlu_classify/` 目录下执行：

```bash
# CPU 训练（自己电脑，速度慢但能跑）
python train_nlu.py --device cpu

# GPU 训练（服务器，推荐）
python train_nlu.py --device cuda

# 自定义参数（CPU 时适当减小 batch_size 加快速度）
python train_nlu.py --device cpu --epochs 10 --batch_size 8
```

训练完成后，`nlu_model/` 目录会自动生成，包含：
- `model.safetensors`：模型权重
- `config.json`：模型配置（含意图分类标签）
- `tokenizer*`：分词器文件
- `label_mapping.json`：意图编号映射

训练完成后可用 HuggingFace 格式直接测试：

```bash
python predict_nlu.py --interactive
```

---

## 推理部署（二选一）

训练完成后，根据需要从以下两条路径中**任选其一**：

| 方式 | 适用场景 | 推理引擎 |
|------|---------|---------|
| 路径 A：GGUF | x86 服务器、桌面端、嵌入式设备 | llama.cpp |
| 路径 B：ONNX | Android 移动端 | ONNX Runtime |

---

### 路径 A：GGUF / llama.cpp

#### 步骤 A1：克隆 llama.cpp

将 llama.cpp 克隆到 `nlu_classify/` 目录内：

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

克隆后目录里会有：
- `llama.cpp/convert_hf_to_gguf.py`：转换脚本（纯 Python，无需编译）
- `llama.cpp/gguf-py/`：Python 的 GGUF 读写库

> **注意**：转换 GGUF 只需要克隆，不需要编译 C++ 代码。

#### 步骤 A2：转换 GGUF

**A2.1 先打补丁（只需执行一次）**

llama.cpp 的转换脚本不认识 bert-base-chinese 的分词器，需要先修复：

```bash
python patch_llama_cpp.py
```

脚本会自动从本地 `bert-base-chinese/` 加载 tokenizer 并动态计算哈希值，
再写入 `llama.cpp/conversion/base.py`。无论使用哪个版本的 llama.cpp 都能正确适配。
成功后会提示 `✅ patch 完成`，同时自动备份原文件为 `base.py.bak`。

**A2.2 选择量化精度**

量化会把权重从 float32 压缩为更少位数，减小文件体积、加快加载速度，但会轻微损失精度。
对意图分类任务影响较小，**移动端部署推荐 Q8_0**。

| 格式 | 文件大小 | 精度影响 | 冷启动速度 | 推荐场景 |
|------|---------|---------|----------|---------|
| F32  | 100%（基准）| 无损 | 最慢 | 开发调试、精度对比基准 |
| F16  | ~50% | 几乎无损（<0.1%） | 快 | 对精度敏感但想省空间 |
| Q8_0 | ~25% | 基本无损（<0.5%） | 很快 | **移动端部署推荐** |
| Q4_K_M | ~12% | 轻微下降（1~3%） | 最快 | 极度资源受限的场景 |

**A2.3 执行转换**

根据需要选择一种精度执行，文件名自动带上精度后缀方便区分：

```bash
# F32（全精度，不量化）
python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype f32 \
    --outfile nlu_model-bert-base-chinese-F32.gguf

# F16（半精度，体积减半，精度几乎不变）
python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype f16 \
    --outfile nlu_model-bert-base-chinese-F16.gguf

# Q8_0（8位量化，推荐）
python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype q8_0 \
    --outfile nlu_model-bert-base-chinese-Q8_0.gguf

# Q4_K_M（4位量化，体积最小）
python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype q4_k_m \
    --outfile nlu_model-bert-base-chinese-Q4_K_M.gguf
```

#### 步骤 A3：追加 pooler 层

llama.cpp 转换时会漏掉 pooler 层，需手动补回。用 `--input` 指定上一步生成的 GGUF 文件，
输出文件名自动在末尾加 `-pooler`（也可用 `--output` 手动指定）：

```bash
# F32
python add_pooler_to_gguf.py \
    --input nlu_model-bert-base-chinese-F32.gguf

# Q8_0（推荐）
python add_pooler_to_gguf.py \
    --input nlu_model-bert-base-chinese-Q8_0.gguf

# 不传参数时默认处理 F32 文件（向后兼容）
python add_pooler_to_gguf.py
```

输出文件示例：`nlu_model-bert-base-chinese-Q8_0-pooler.gguf`

#### 步骤 A4：GGUF 推理测试

交互式脚本需先激活环境，不能用 `conda run`：

```bash
# 默认加载 F32 版本
python test_gguf.py

# 指定量化版本（用 --model 参数）
python test_gguf.py --model nlu_model-bert-base-chinese-Q8_0-pooler.gguf
```

测试示例：
```
话语 > 打开空调
  预测：直接车控  （置信度：97.3%）
  Top-3 候选：
    1. 直接车控      97.3%  ████████████████████████
    2. 感知车控       1.8%
    3. 闲聊           0.9%
```

#### 附：x86 编译 llama.cpp（可选）

如果将来需要使用 llama.cpp 的 C++ 原生工具（如 `llama-cli` 命令行推理），才需要编译。纯 Python 工作流不需要这一步。

**Linux x86（CPU only）：**

```bash
cd llama.cpp

# 方式一：cmake（推荐）
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j$(nproc)

# 方式二：make
make -j$(nproc)

cd ..
```

编译完成后，`llama.cpp/build/bin/` 下会有 `llama-cli`、`llama-embedding` 等工具。

**如果服务器有 GPU（CUDA）：**

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

---

### 路径 B：ONNX（Android 移动端部署）

如果你需要在 Android 应用中使用 ONNX Runtime 进行推理（而非 llama.cpp），
可直接从训练好的 `nlu_model/` 一键导出 ONNX 模型，包含量化、tokenizer 和后处理。

```bash
python convert_to_onnx.py
```

脚本分三步执行：

| 步骤 | 操作 | 产出 | 说明 |
|------|------|------|------|
| 1 | HuggingFace → ONNX | `nlu_model.onnx` | FP32 原始导出，409 MB |
| 2 | INT8 动态量化 | `nlu_model_quant.onnx` | 权重 INT8，103 MB（-75%） |
| 3 | 集成 pre/post processing | `nlu_model_mobile.onnx` | 量化 + BertTokenizer + argmax + label 映射 |

最终产物 `nlu_model_mobile.onnx` 内置了 tokenizer，在 Android 端只需一行调用：

```kotlin
// 输入纯文本，输出分类索引
session.run("input_text", "打开空调")  // → index = 4 → "直接车控"
```

> **注意**：`convert_to_onnx.py` 不依赖 llama.cpp，无需先执行路径 A 的步骤。从 `nlu_model/` 直接导出即可。
> 导出需要额外依赖：`onnxruntime`、`onnx`、`onnxruntime-extensions`、`onnxscript`，`setup_env.sh` 已包含这些检查。

---

## 完整流程一览

```
训练数据 (training_data/*.csv)
    ↓
python train_nlu.py                    → nlu_model/（HuggingFace 格式）
    ↓
    ├─ python predict_nlu.py           → 直接测试（开发阶段用）
    │
    ├─ [路径 A：GGUF / llama.cpp] ────────────────────┐
    │                                                    │
    │   python llama.cpp/convert_hf_to_gguf.py           │
    │       → nlu_model-bert-base-chinese-{精度}.gguf    │
    │   python add_pooler_to_gguf.py --input …           │
    │       → nlu_model-bert-base-chinese-{精度}-pooler.gguf │
    │   python test_gguf.py             → GGUF 推理测试  │
    │                                                    │
    └─ [路径 B：ONNX / Android] ────────────────────────┘
                                                        │
        python convert_to_onnx.py      → nlu_model_onnx/
            nlu_model_mobile.onnx      → Android 直接加载
```
