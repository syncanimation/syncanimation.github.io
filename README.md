# SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation 

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)  
[Project Page](https://syncanimation.github.io/) | [Paper (arXiv)](https://arxiv.org/abs/2501.14646)   

> “Generating talking avatar driven by audio remains a significant challenge. Existing methods typically require high computational costs and often lack sufficient facial detail and realism, making them unsuitable for applications that demand high real-time performance and visual quality. Additionally, while some methods can synchronize lip movement, they still face issues with consistency between facial expressions and upper body movement, particularly during silent periods. In this paper, we introduce SyncAnimation, the first NeRF-based method that achieves audio-driven, stable, and real-time generation of speaking avatar by combining generalized audio-to-pose matching and audio-to-expression synchronization. By integrating AudioPose Syncer and AudioEmotion Syncer, SyncAnimation achieves high-precision poses and expression generation, progressively producing audio-synchronized upper body, head, and lip shapes. Furthermore, the High-Synchronization Human Renderer ensures seamless integration of the head and upper body, and achieves audio-sync lip.”  

<!--
## 🧠 简介  

语音驱动的人脸合成在姿态、同步性、细节还原等方面存在挑战。**SyncTalk** 的核心在于**同步性**（lip sync + 表情 + 头部运动一致性）控制。为了解决这些问题，SyncTalk 采用以下机制：

- 基于三平面哈希 (tri-plane hash) 表示来保持身份一致性  
- 同时生成同步口型、面部表情和稳定头部姿态  
- 在高分辨率视频中恢复头发等细节  
-->

## 🛠 安装与依赖  

### Linux / Ubuntu  

以下为在 Ubuntu 上的推荐安装流程（已知在 Ubuntu 18.04 + PyTorch 1.12.1 + CUDA 11.3 下进行测试）：

```bash
git clone https://github.com/ZiqiaoPeng/SyncTalk.git
cd SyncTalk

# 建议使用 conda 环境
conda create -n synctalk python==3.8.8
conda activate synctalk

# 安装 PyTorch 与 torchvision（根据 CUDA 版本选择对应版本）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

sudo apt-get install portaudio19-dev
pip install -r requirements.txt

# 安装依赖模块（freqencoder / gridencoder / shencoder / raymarching）
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching

# 安装 PyTorch3D（若有困难，可使用脚本 fallback 方案）
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
# 或者：
python ./scripts/install_pytorch3d.py

# 安装 TensorFlow GPU 版本（部分模块可能依赖）
pip install tensorflow-gpu==2.8.1
```

> **提示**：安装 PyTorch3D 时可能遇到兼容性问题，建议优先使用 `scripts/install_pytorch3d.py` 处理。

---

<!--
## 🔄 数据准备  

### 预训练模型  

请将下载的预训练模型（例如 `May.zip`、`trial_may.zip`）放入相应目录：

- `data/May.zip` → 解压至 `data/May/`  
- `model/trial_may.zip` → 解压至 `model/trial_may/`  

### 输入视频处理  

1. 下载 face-parsing 模型  
   ```bash
   wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
   ```
2. 下载 3DMM 头姿估计模型  
   ```bash
   wget …  # 多个文件：exp_info.npy, keys_info.npy, sub_mesh.obj, topology_info.npy  
   ```
3. 下载 Basel Face Model (BFM 2009)，转换为内部格式  
   ```bash
   # 下载 .mat 模型文件，并放到 data_utils/face_tracking/3DMM/
   cd data_utils/face_tracking
   python convert_BFM.py
   ```
4. 输入视频要求：  
   - 帧率 25 FPS  
   - 每帧包含讲话人面部  
   - 分辨率推荐 ~512×512  
   - 时长约 4–5 分钟  

5. 执行视频处理脚本  
   ```bash
   python data_utils/process.py data/<ID>/<ID>.mp4 --asr ave
   ```
   - 支持 `ave`、`deepspeech`、`hubert` 三种特征提取  
   - 可选地，运行 OpenFace 的 `FeatureExtraction`，生成眼睛眨动 AU45 信息（重命名为 `data/<ID>/au.csv`）  

> 注意：由于 EmoTalk 的 blendshape 捕捉未开源，因此这里默认使用 mediapipe 的 blendshape 捕捉。对于某些场景效果不佳，可考虑替换或改进。  

---

## 🚀 快速上手  

### 评估 / 推理  

```bash
python main.py data/May --workspace model/trial_may -O --test --asr_model ave
python main.py data/May --workspace model/trial_may -O --test --asr_model ave --portrait
```

- `--portrait`：将生成的人脸贴回原始图像 → 画质较好  
- 成功运行后将输出 PSNR / LPIPS / LMD 等指标  

若要使用目标音频进行推理：

```bash
python main.py data/May --workspace model/trial_may -O --test --test_train --asr_model ave --portrait --aud ./demo/test.wav
```

注意：音频需为 `.wav` 格式。如果使用其他特征（如 npy），可改路径至对应文件。

### 训练  

默认方式从磁盘按需加载数据：

```bash
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model ave
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model ave
# 或者使用脚本
sh ./scripts/train_may.sh
```

若想加入躯干训练以修复双下巴（注意：训练躯干后不能用 `--portrait` 模式）：

```bash
python main.py data/May/ --workspace model/trial_may_torso/ -O --torso --head_ckpt <head_ckpt>.pth --iters 150000 --asr_model ave
python main.py data/May --workspace model/trial_may_torso -O --torso --test --asr_model ave
python main.py data/May --workspace model/trial_may_torso -O --torso --test --test_train --asr_model ave --aud ./demo/test.wav
```

---

## 📊 评价指标  

示例（单人）：

| 模式                      | PSNR    | LPIPS    | LMD     |
|---------------------------|---------|-----------|---------|
| SyncTalk (不贴回原图)     | 32.201  | 0.0394    | 2.822   |
| SyncTalk (贴回原图)       | 37.644  | 0.0117    | 2.825   |

（论文给出了多个被试的平均指标）

---
 -->

## 📝 Citation  

Please cite the following paper if you use this method, model, or conduct derivative research based on this project:

```tex
@inproceedings{ijcai2025p185,
  title     = {SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation},
  author    = {Liu, Yujian and Xu, Shidang and Guo, Jing and Wang, Dingbin and Wang, Zairan and Tan, Xianfeng and Liu, Xiaoli},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {1657--1665},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/185},
  url       = {https://doi.org/10.24963/ijcai.2025/185},
}
```

---

## 🙏 Acknowledgements  

This project is built upon or inspired by the following open-source projects:

- Synctalk  
- ER-NeRF  
- GeneFace   
- AD-NeRF  
- Deep3DFaceRecon_pytorch  

We sincerely thank the authors of these projects for their contributions to the open-source community.

---

## ⚠️ Disclaimer  

By using this project, you agree to comply with all applicable laws and regulations.
You must not use it to generate or disseminate harmful content.
The developers assume no responsibility for any direct, indirect, or consequential damages arising from the use or misuse of this software. 
