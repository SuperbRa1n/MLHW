import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import clip
from torchvision import transforms
from torchvision.datasets import ImageNet
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt


# 设置随机种子保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


class CLIPAdapter(nn.Module):
    """CLIP-Adapter: 添加轻量级适配器到CLIP模型"""

    def __init__(self, clip_model, adapter_dim=64, alpha=0.2):
        """
        Args:
            clip_model: 预训练的CLIP模型
            adapter_dim: 适配器中间层维度
            alpha: 残差连接权重
        """
        super().__init__()
        self.clip_model = clip_model
        self.alpha = alpha

        # 冻结CLIP的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 图像适配器 (添加到图像编码器后)
        input_dim = clip_model.visual.output_dim
        self.image_adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, input_dim)
        )

        # 文本适配器 (添加到文本编码器后)
        self.text_adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, input_dim)
        )

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        """
        Args:
            images: 输入图像
            texts: 输入文本标记
        Returns:
            image_features: 图像特征
            text_features: 文本特征
        """
        # 获取原始CLIP特征
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)

        # 应用适配器
        adapted_image = self.image_adapter(image_features)
        adapted_text = self.text_adapter(text_features)

        # 残差连接
        image_features = image_features + self.alpha * adapted_image
        text_features = text_features + self.alpha * adapted_text

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

    def encode_image(self, images):
        """仅编码图像"""
        image_features = self.clip_model.encode_image(images)
        adapted_image = self.image_adapter(image_features)
        image_features = image_features + self.alpha * adapted_image
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        """仅编码文本"""
        text_features = self.clip_model.encode_text(texts)
        adapted_text = self.text_adapter(text_features)
        text_features = text_features + self.alpha * adapted_text
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


class FewShotImageNet:
    """创建few-shot ImageNet数据集"""

    def __init__(self, root, split='train', num_shots=16, transform=None):
        """
        Args:
            root: ImageNet根目录
            split: 'train'或'val'
            num_shots: 每个类别的样本数
            transform: 数据增强
        """
        self.dataset = ImageNet(root=root, split=split, transform=transform)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        self.num_shots = num_shots

        # 为每个类别选择固定数量的样本
        self.indices = self._create_few_shot_indices()

    def _create_few_shot_indices(self):
        """为每个类别选择固定数量的样本"""
        # 按类别组织样本索引
        class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        # 为每个类别选择固定数量的样本
        selected_indices = []
        for label, indices in class_to_indices.items():
            # 如果该类别的样本少于num_shots，使用所有样本
            n_samples = min(self.num_shots, len(indices))
            selected = np.random.choice(indices, n_samples, replace=False)
            selected_indices.extend(selected.tolist())

        return selected_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]


def prepare_text_prompts(class_names, template="a photo of a {}"):
    """准备文本提示"""
    text_inputs = torch.cat([
        clip.tokenize(template.format(c)) for c in class_names
    ])
    return text_inputs


def train_adapter(model, train_loader, val_loader, text_prompts, args):
    """训练适配器"""
    device = next(model.parameters()).device

    # 只训练适配器参数
    trainable_params = []
    for name, param in model.named_parameters():
        if 'adapter' in name or 'logit_scale' in name:
            trainable_params.append(param)
            print(f"Training parameter: {name}")

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = text_prompts.to(device)

            # 前向传播
            image_features, text_features = model(images, text_inputs)

            # 计算logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            # 计算损失
            loss = criterion(logits_per_image, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        val_acc = evaluate(model, val_loader, text_prompts, args)
        val_accuracies.append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'best_adapter_{args.num_shots}shot.pth')

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig(f'training_curves_{args.num_shots}shot.png')
    plt.close()

    return best_acc


def evaluate(model, data_loader, text_prompts, args, topk=(1, 5)):
    """评估模型性能"""
    model.eval()
    device = next(model.parameters()).device

    # 预计算文本特征
    with torch.no_grad():
        text_features = model.encode_text(text_prompts.to(device))

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # 获取图像特征
            image_features = model.encode_image(images)

            # 计算相似度
            logit_scale = model.logit_scale.exp()
            similarity = logit_scale * image_features @ text_features.t()

            # 计算top-1和top-5准确率
            _, pred_top5 = similarity.topk(5, dim=1)
            _, pred_top1 = similarity.topk(1, dim=1)

            # 转换为与labels相同的形状
            pred_top1 = pred_top1.squeeze()
            if pred_top1.dim() == 0:
                pred_top1 = pred_top1.unsqueeze(0)

            # 统计正确预测
            correct_top1 += (pred_top1 == labels).sum().item()
            correct_top5 += (pred_top5 == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    return top1_acc


def analyze_predictions(model, data_loader, text_prompts, class_names, args, num_samples=5):
    """分析预测结果，展示一些例子"""
    model.eval()
    device = next(model.parameters()).device

    # 预计算文本特征
    with torch.no_grad():
        text_features = model.encode_text(text_prompts.to(device))

    # 获取一些样本进行分析
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        images, labels = next(data_iter)
        images = images.to(device)
        labels = labels.to(device)

        # 获取预测
        with torch.no_grad():
            image_features = model.encode_image(images)
            logit_scale = model.logit_scale.exp()
            similarity = logit_scale * image_features @ text_features.t()
            probs = similarity.softmax(dim=-1)

            # 获取top-3预测
            top3_probs, top3_indices = probs[0].topk(3)

        # 显示图像
        img = images[0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {class_names[labels[0].item()]}")
        axes[i, 0].axis('off')

        # 显示预测概率
        axes[i, 1].barh(range(3), top3_probs.cpu().numpy())
        axes[i, 1].set_yticks(range(3))
        axes[i, 1].set_yticklabels([class_names[idx] for idx in top3_indices.cpu().numpy()])
        axes[i, 1].set_xlabel('Probability')
        axes[i, 1].set_title('Top-3 Predictions')
        axes[i, 1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'prediction_analysis_{args.num_shots}shot.png')
    plt.show()

    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 50}")
    print("Parameter Statistics:")
    print(f"{'=' * 50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description='CLIP-Adapter for Few-shot ImageNet')
    parser.add_argument('--data_root', type=str, default='./data/ImageNet-Mini', help='ImageNet data root')
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots per class')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--adapter_dim', type=int, default=64, help='Adapter dimension')
    parser.add_argument('--alpha', type=float, default=0.2, help='Adapter residual weight')
    parser.add_argument('--clip_backbone', type=str, default='RN50', help='CLIP backbone')
    parser.add_argument('--num_val_samples', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"CLIP backbone: {args.clip_backbone}")
    print(f"Few-shot setting: {args.num_shots}-shot")

    # 1. 加载CLIP模型
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(args.clip_backbone, device=args.device)

    # 2. 创建CLIP-Adapter模型
    print("Creating CLIP-Adapter...")
    model = CLIPAdapter(clip_model, adapter_dim=args.adapter_dim, alpha=args.alpha)
    model = model.to(args.device)

    # 3. 准备数据集
    print("Preparing datasets...")

    # 训练集：few-shot ImageNet
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        preprocess
    ])

    train_dataset = FewShotImageNet(
        root=args.data_root,
        split='train',
        num_shots=args.num_shots,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 验证集：使用部分ImageNet验证集（加速评估）
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess
    ])

    full_val_dataset = ImageNet(root=args.data_root, split='val', transform=val_transform)

    # 随机选择部分验证样本
    val_indices = np.random.choice(
        len(full_val_dataset),
        min(args.num_val_samples, len(full_val_dataset)),
        replace=False
    )

    val_dataset = Subset(full_val_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 获取类别名称
    class_names = train_dataset.classes

    # 4. 准备文本提示
    print("Preparing text prompts...")
    text_prompts = prepare_text_prompts(class_names)

    # 5. 评估原始CLIP的zero-shot性能
    print("\n" + "=" * 50)
    print("Evaluating original CLIP (zero-shot)...")
    print("=" * 50)

    # 临时使用原始CLIP进行评估
    original_clip_model = clip_model
    original_text_features = original_clip_model.encode_text(text_prompts.to(args.device))
    original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

    # 快速评估
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            image_features = original_clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ original_text_features.t()).softmax(dim=-1)
            predictions = similarity.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    original_acc = 100.0 * correct / total
    print(f"Original CLIP zero-shot accuracy: {original_acc:.2f}%")

    # 6. 训练适配器
    print("\n" + "=" * 50)
    print(f"Training CLIP-Adapter ({args.num_shots}-shot)...")
    print("=" * 50)

    best_acc = train_adapter(model, train_loader, val_loader, text_prompts, args)

    # 7. 加载最佳模型并评估
    print("\n" + "=" * 50)
    print("Evaluating best model...")
    print("=" * 50)

    checkpoint = torch.load(f'best_adapter_{args.num_shots}shot.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_acc = evaluate(model, val_loader, text_prompts, args)

    # 8. 分析预测结果
    print("\n" + "=" * 50)
    print("Analyzing predictions...")
    print("=" * 50)

    analyze_predictions(model, val_loader, text_prompts, class_names, args)

    # 9. 打印总结
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"CLIP Backbone: {args.clip_backbone}")
    print(f"Few-shot setting: {args.num_shots}-shot per class")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Adapter dimension: {args.adapter_dim}")
    print(f"Adapter alpha: {args.alpha}")
    print(f"Original CLIP zero-shot accuracy: {original_acc:.2f}%")
    print(f"CLIP-Adapter few-shot accuracy: {final_acc:.2f}%")
    print(f"Absolute improvement: {final_acc - original_acc:.2f}%")
    print(f"Relative improvement: {100 * (final_acc - original_acc) / original_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()