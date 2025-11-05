# ⚙️ Setup NVIDIA GPU for Deep Learning

This repository provides a simple guide to set up your **NVIDIA GPU** for Deep Learning using **PyTorch** or **TensorFlow**.

> 🔗 **Reference:** [Original Guide by entbappy](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)

---

## 🧩 Steps Overview

### 1️⃣ Install NVIDIA Driver  
👉 [Download Here](https://www.nvidia.com/Download/index.aspx)  
Make sure to install the latest driver for your GPU model.  

**Test installation:**
```bash
nvidia-smi
```

---

### 2️⃣ Install Visual Studio (C++)  
👉 [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)  
Select “**Desktop Development with C++**” during installation.

---

### 3️⃣ Install Anaconda / Miniconda  
👉 [Download Anaconda](https://www.anaconda.com/download/success)  
Create a new environment:
```bash
conda create -n dl_env python=3.10
conda activate dl_env
```

---

### 4️⃣ Install CUDA Toolkit  
👉 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)  
Choose a version compatible with your PyTorch/TensorFlow release.  

**Verify installation:**
```bash
nvcc --version
```

---

### 5️⃣ Install cuDNN  
👉 [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)  
Extract and copy the files into your CUDA installation directories:  
```
bin → CUDA\bin  
include → CUDA\include  
lib → CUDA\lib
```

---

### 6️⃣ Install PyTorch (GPU Enabled)  
👉 [Install PyTorch](https://pytorch.org/get-started/locally/)  

Example command:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## ✅ Verify GPU Setup
Run this Python script to test your GPU:
```python
import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only")
```

**Expected Output:**
```
GPU Available: True
GPU Name: NVIDIA GeForce RTX XXXX
```

---

## 🔗 Useful Links
- [NVIDIA Developer Portal](https://developer.nvidia.com/)
- [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

---

### 🧠 Credits
Based on: [entbappy/Setup-NVIDIA-GPU-for-Deep-Learning](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)  
Updated and refined by **[Your Name](https://github.com/your-github-username)** ✨  

> 💡 *Your GPU is now ready for deep learning workloads!*
