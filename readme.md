# NLU 意图分类 · 完整流程

基于 BERT 的车载语音意图分类，支持训练、HuggingFace 推理、导出 GGUF 嵌入式推理。

---

## 目录结构

```
nlu_classify/
├── train_nlu.py                               # 训练脚本
├── predict_nlu.py                             # HuggingFace 格式推理
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
├── llama.cpp/                                 # 需手动克隆（见步骤三）
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

所有 Python 命令都需要在 conda 环境中执行。以下文档用 `<ENV>` 代替环境名，
**请将 `<ENV>` 替换成你自己的环境名**。

**如果已有现成环境**（里面有 torch、transformers），直接在里面安装缺少的包：

```bash
conda activate <ENV>
pip install llama-cpp-python safetensors openpyxl
```

**如果从零新建环境**：

```bash
# 创建新环境（Python 3.10）
conda create -n <ENV> python=3.10 -y
conda activate <ENV>

# 安装依赖
pip install torch transformers scikit-learn pandas openpyxl
pip install llama-cpp-python safetensors
```

---

## 步骤二：训练

在 `nlu_classify/` 目录下执行：

```bash
# CPU 训练（自己电脑，速度慢但能跑）
conda run -n <ENV> python train_nlu.py --device cpu

# GPU 训练（服务器，推荐）
conda run -n <ENV> python train_nlu.py --device cuda

# 自定义参数（CPU 时适当减小 batch_size 加快速度）
conda run -n <ENV> python train_nlu.py --device cpu --epochs 10 --batch_size 8
```

训练完成后，`nlu_model/` 目录会自动生成，包含：
- `model.safetensors`：模型权重
- `config.json`：模型配置（含意图分类标签）
- `tokenizer*`：分词器文件
- `label_mapping.json`：意图编号映射

训练完成后可用 HuggingFace 格式直接测试：

```bash
# 交互式脚本需要先激活环境再运行，不能用 conda run（会断开键盘输入）
conda activate <ENV>
python predict_nlu.py --interactive
```

---

## 步骤三：克隆 llama.cpp

将 llama.cpp 克隆到 `nlu_classify/` 目录内：

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

克隆后目录里会有：
- `llama.cpp/convert_hf_to_gguf.py`：转换脚本（纯 Python，无需编译）
- `llama.cpp/gguf-py/`：Python 的 GGUF 读写库

> **注意**：转换 GGUF 只需要克隆，不需要编译 C++ 代码。

---

## 步骤四：转换 GGUF

**4.1 先打补丁（只需执行一次）**

llama.cpp 的转换脚本不认识 bert-base-chinese 的分词器，需要先修复：

```bash
conda run -n <ENV> python patch_llama_cpp.py
```

脚本会自动从本地 `bert-base-chinese/` 加载 tokenizer 并动态计算哈希值，
再写入 `llama.cpp/conversion/base.py`。无论使用哪个版本的 llama.cpp 都能正确适配。
成功后会提示 `✅ patch 完成`，同时自动备份原文件为 `base.py.bak`。

**4.2 选择量化精度**

量化会把权重从 float32 压缩为更少位数，减小文件体积、加快加载速度，但会轻微损失精度。
对意图分类任务影响较小，**移动端部署推荐 Q8_0**。

| 格式 | 文件大小 | 精度影响 | 冷启动速度 | 推荐场景 |
|------|---------|---------|----------|---------|
| F32  | 100%（基准）| 无损 | 最慢 | 开发调试、精度对比基准 |
| F16  | ~50% | 几乎无损（<0.1%） | 快 | 对精度敏感但想省空间 |
| Q8_0 | ~25% | 基本无损（<0.5%） | 很快 | **移动端部署推荐** |
| Q4_K_M | ~12% | 轻微下降（1~3%） | 最快 | 极度资源受限的场景 |

**4.3 执行转换**

根据需要选择一种精度执行，文件名自动带上精度后缀方便区分：

```bash
# F32（全精度，不量化）
conda run -n <ENV> python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype f32 \
    --outfile nlu_model-bert-base-chinese-F32.gguf

# F16（半精度，体积减半，精度几乎不变）
conda run -n <ENV> python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype f16 \
    --outfile nlu_model-bert-base-chinese-F16.gguf

# Q8_0（8位量化，推荐）
conda run -n <ENV> python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype q8_0 \
    --outfile nlu_model-bert-base-chinese-Q8_0.gguf

# Q4_K_M（4位量化，体积最小）
conda run -n <ENV> python llama.cpp/convert_hf_to_gguf.py \
    nlu_model --outtype q4_k_m \
    --outfile nlu_model-bert-base-chinese-Q4_K_M.gguf
```

---

## 步骤五：追加 pooler 层

llama.cpp 转换时会漏掉 pooler 层，需手动补回。用 `--input` 指定上一步生成的 GGUF 文件，
输出文件名自动在末尾加 `-pooler`（也可用 `--output` 手动指定）：

```bash
# F32
conda run -n <ENV> python add_pooler_to_gguf.py \
    --input nlu_model-bert-base-chinese-F32.gguf

# Q8_0（推荐）
conda run -n <ENV> python add_pooler_to_gguf.py \
    --input nlu_model-bert-base-chinese-Q8_0.gguf

# 不传参数时默认处理 F32 文件（向后兼容）
conda run -n <ENV> python add_pooler_to_gguf.py
```

输出文件示例：`nlu_model-bert-base-chinese-Q8_0-pooler.gguf`

---

## 步骤六：GGUF 推理测试

`test_gguf.py` 默认加载 `nlu_model-bert-base-chinese-F32-pooler.gguf`，
如果用的是其他量化版本，修改脚本第 31 行的 `GGUF_PATH` 变量即可：

```python
GGUF_PATH = "nlu_model-bert-base-chinese-Q8_0-pooler.gguf"
```

然后运行（交互式脚本需先激活环境，不能用 `conda run`）：

```bash
conda activate <ENV>
python test_gguf.py
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

---

## 附：x86 编译 llama.cpp（可选）

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

## 完整流程一览

```
训练数据 (training_data/*.csv)
    ↓
python train_nlu.py                    → nlu_model/（HuggingFace 格式）
    ↓
python predict_nlu.py                  → 直接测试（开发阶段用）
    ↓
python llama.cpp/convert_hf_to_gguf.py → nlu_model-bert-base-chinese-{精度}.gguf
    （选择 f32 / f16 / q8_0 / q4_k_m）
    ↓
python add_pooler_to_gguf.py --input … → nlu_model-bert-base-chinese-{精度}-pooler.gguf
    ↓
python test_gguf.py                    → GGUF 推理测试（部署阶段用）
```
