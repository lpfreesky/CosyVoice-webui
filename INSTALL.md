# CosyVoice2-0.5B 安装教程

## 环境要求

- **系统**: Windows 10/11
- **GPU**: NVIDIA RTX 4050 6GB (或其他支持 CUDA 的显卡)
- **Python**: 3.11
- **CUDA**: 12.1 (注意：不是 12.4)

---

## 一、环境准备

### 1. 安装 Python 3.11

建议使用 Anaconda:
```bash
conda create -n cosyvoice python=3.11
conda activate cosyvoice
```

### 2. 安装 CUDA 12.1

下载并安装 [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64)

---

## 二、依赖安装

### 1. 克隆项目

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
```

### 2. 安装 PyTorch (关键！)

**必须使用 PyTorch 2.3.1 + CUDA 12.1**

```bash
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

**⚠️ 重要提醒**：
- 不要使用 PyTorch 2.6.0 或更高版本，会导致模型输出杂音
- 不要使用 CUDA 12.4 版本，必须用 cu121

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

或者手动安装关键依赖:

```bash
pip install transformers==4.51.3
pip install onnxruntime==1.18.0
pip install gradio==5.4.0
pip install librosa==0.10.2
pip install soundfile==0.12.1
pip install torchaudio==2.3.1
```

---

## 三、模型下载

### 1. 下载 CosyVoice2-0.5B 模型

从 HuggingFace 下载:

```python
from modelscope import snapshot_download
model_dir = snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
```

或者使用 huggingface-cli:

```bash
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir pretrained_models/CosyVoice2-0.5B
```

### 2. 模型文件列表

下载后应有以下文件 (总大小约 2GB):

```
pretrained_models/CosyVoice2-0.5B/
├── .gitattributes
├── campplus.onnx                    #说话人提取模型
├── config.json
├── configuration.json
├── cosyvoice2.yaml
├── flow.decoder.estimator.fp32.onnx #流式合成模型
├── flow.pt                          #Flow 模型 (~400MB)
├── hift.pt                          #HiFT 模型 (~100MB)
├── llm.pt                           #LLM 模型 (~1GB)
├── README.md
├── speech_tokenizer_v2.batch.onnx
├── speech_tokenizer_v2.onnx
├── asset/
│   └── dingding.png
└── CosyVoice-BlankEN/
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model.safetensors             #主模型 (~900MB)
    ├── tokenizer_config.json
    ├── vocab.json
    └── tokenizer.json
```

---

## 四、运行 WebUI

### 1. 启动命令

```bash
cd CosyVoice
python webui.py
```

### 2. 访问地址

打开浏览器访问: http://localhost:8000

---

## 五、使用说明

### 3s极速复刻模式

1. **上传录音**: 上传 1-30 秒的人声录音
2. **填写文本**: 输入录音中实际说的内容
3. **填写合成文本**: 输入要合成的文本
4. **生成**: 点击生成按钮

### 音频格式要求

- **格式**: WAV (推荐) 或 MP3/M4A
- **采样率**: 24kHz (如使用其他格式，程序会自动转换)
- **时长**: 1-30 秒
- **内容**: 清晰的人声

### 转换录音格式

如需转换格式:
```bash
ffmpeg -i "录音.m4a" -ar 24000 -ac 1 prompt.wav
```

参数说明:
- `-ar 24000`: 采样率 24kHz
- `-ac 1`: 单声道

---

## 六、已知问题

### 问题: 输出"哦哦啊啊"杂音

**原因**: PyTorch 版本不兼容

**解决方案**:
```bash
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3
```

---

## 七、最终依赖版本汇总

```
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
torchvision==0.18.1+cu121
transformers==4.51.3
onnxruntime==1.18.0
gradio==5.4.0
librosa==0.10.2
soundfile==0.12.1
numpy==1.26.4
```

---

## 八、参考链接

- GitHub: https://github.com/FunAudioLLM/CosyVoice
- HuggingFace: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- ModelScope: https://modelscope.cn/models/AI-ModelScope/CosyVoice2-0.5B
