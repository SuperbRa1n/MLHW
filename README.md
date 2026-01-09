# CLIP-Adapter: Few-Shot Learning with CLIP

本项目基于 [OpenAI CLIP](https://github.com/openai/CLIP) 仓库修改，主要实现了 **CLIP-Adapter** 方法，用于在 ImageNet 数据集上进行少样本学习（Few-Shot Learning）。

## 项目简介

本项目在原始 CLIP 模型的基础上，添加了轻量级的适配器（Adapter）模块，通过冻结 CLIP 的预训练参数，只训练少量可学习的适配器参数，实现在少量样本上的高效微调。相比全参数微调，CLIP-Adapter 具有以下优势：

- **参数效率高**：只训练少量适配器参数，大幅降低训练成本
- **训练速度快**：冻结 CLIP 主干网络，训练速度更快
- **性能提升明显**：在 few-shot 设置下相比 zero-shot 有显著提升

## 快速开始

### 1. 环境安装

首先安装必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

运行下载脚本准备数据（详见下方"数据集下载"章节）：

```bash
python download_dataset.py
```

### 3. 训练模型

使用 16-shot 设置训练 CLIP-Adapter：

```bash
python Adapter_16shot.py --data_root ./data/ImageNet-Custom --num_shots 16
```

## 原始 CLIP 介绍

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet "zero-shot" without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)

## 数据集下载

本项目使用 ImageNet 数据集进行训练和评估。我们提供了一个自动下载脚本 `download_dataset.py` 来下载和准备数据。

### 下载步骤

1. **配置 Hugging Face Token**

   由于 ImageNet 是受限数据集，需要先获取 Hugging Face Token：
   - 访问 https://huggingface.co/settings/tokens 申请 Token
   - 在 ImageNet 官网同意使用协议
   - 编辑 `download_dataset.py`，将 `HF_TOKEN` 变量设置为你的 Token

2. **运行下载脚本**

```bash
python download_dataset.py
```

脚本会自动完成以下操作：
- 下载 ImageNet 类别映射文件（`imagenet_class_index.json`）
- 从 Hugging Face 下载 ImageNet 数据集（默认每类 32 张图片）
- 自动划分训练集和验证集（默认验证集占比 50%）
- 将数据保存到 `./data/ImageNet-Custom/` 目录

下载完成后，数据目录结构如下：
```
data/ImageNet-Custom/
├── train/          # 训练集
│   ├── n01440764/  # 类别文件夹
│   ├── n01443537/
│   └── ...
├── val/            # 验证集
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── imagenet_class_index.json  # 类别映射文件
```

### 配置参数

在 `download_dataset.py` 中可以调整以下参数：
- `TOTAL_IMAGES_PER_CLASS`: 每个类别下载的图片总数（默认 32）
- `VAL_RATIO`: 验证集占比（默认 0.5，即 50%）
- `ROOT_DIR`: 数据保存路径（默认 `./data/ImageNet-Custom`）

## 运行 CLIP-Adapter

本项目提供了两个训练脚本，分别适用于不同的场景：

### 1. Adapter_16shot.py - 16-shot 训练（推荐）

该脚本使用自定义的 ImageNet 数据集（通过 `download_dataset.py` 下载），适合快速实验和测试。

**运行命令：**

```bash
python Adapter_16shot.py \
    --data_root ./data/ImageNet-Custom \
    --num_shots 16 \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-3 \
    --adapter_dim 64 \
    --alpha 0.2 \
    --clip_backbone ViT-B/32
```

**主要参数说明：**
- `--data_root`: ImageNet 数据根目录（默认 `./data/ImageNet-Custom`）
- `--num_shots`: 每个类别的训练样本数（默认 16）
- `--batch_size`: 批次大小（默认 32）
- `--epochs`: 训练轮数（默认 20）
- `--lr`: 学习率（默认 1e-3）
- `--adapter_dim`: 适配器中间层维度（默认 64）
- `--alpha`: 适配器残差连接权重（默认 0.2）
- `--clip_backbone`: CLIP 模型架构（默认 `ViT-B/32`，可选 `RN50`, `ViT-B/16` 等）

**输出文件：**
- `best_adapter_16shot.pth`: 最佳模型 checkpoint
- `training_curves_16shot.png`: 训练曲线图
- `prediction_analysis_16shot.png`: 预测结果分析图

### 2. Adapter_all.py - 完整数据集测试

该脚本使用完整的 ImageNet 数据集（需要预先下载完整 ImageNet）进行测试（但是训练仍使用16shot），适合完整实验。

**运行命令：**

```bash
python Adapter_all.py \
    --data_root ./data/ImageNet-Mini \
    --num_shots 16 \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-3 \
    --adapter_dim 64 \
    --alpha 0.2 \
    --clip_backbone RN50
```

**注意：** 使用该脚本需要先下载完整的 ImageNet 数据集，并确保数据目录结构正确。
