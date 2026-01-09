import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import clip
from torchvision import transforms, datasets  # [修改] 导入 datasets
import json  # [修改] 导入 json 用于读取类别映射
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
        # 获取原始CLIP特征
        # [修改] CLIP 默认输出 float16，这里必须转为 float32 才能进入 Adapter
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(texts)

        image_features = image_features.float()
        text_features = text_features.float()

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
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        image_features = image_features.float()  # [修改] 转 float32

        adapted_image = self.image_adapter(image_features)
        image_features = image_features + self.alpha * adapted_image
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        """仅编码文本"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text(texts)
        text_features = text_features.float()  # [修改] 转 float32

        adapted_text = self.text_adapter(text_features)
        text_features = text_features + self.alpha * adapted_text
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


class FewShotImageNet:
    """创建few-shot ImageNet数据集"""

    def __init__(self, root, split='train', num_shots=16, transform=None):
        # [修改] 使用 ImageFolder 替代 ImageNet 类
        # 拼接路径，例如 ./data/ImageNet-Mini/train
        target_dir = os.path.join(root, split)
        if not os.path.exists(target_dir):
            # 兼容某些只有 root 目录的情况
            if os.path.exists(os.path.join(root, 'n01440764')):
                target_dir = root
            else:
                raise RuntimeError(f"数据目录不存在: {target_dir}")

        self.dataset = datasets.ImageFolder(root=target_dir, transform=transform)

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
            if n_samples > 0:
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
    print(f"Sample prompts: {class_names[:3]}...")
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
            # 注意：如果类别数少于5，topk(5)会报错，加个判断
            k = min(5, similarity.shape[1])
            _, pred_topk = similarity.topk(k, dim=1)
            _, pred_top1 = similarity.topk(1, dim=1)

            # 转换为与labels相同的形状
            pred_top1 = pred_top1.squeeze()
            if pred_top1.dim() == 0:
                pred_top1 = pred_top1.unsqueeze(0)

            # 统计正确预测
            correct_top1 += (pred_top1 == labels).sum().item()
            if k >= 5:
                correct_top5 += (pred_topk == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total if k >= 5 else 0.0

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    if k >= 5:
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
    try:
        images, labels = next(data_iter)
    except StopIteration:
        return

    # 防止 num_samples 大于 batch_size
    actual_samples = min(num_samples, len(images))
    images = images.to(device)
    labels = labels.to(device)

    # 获取预测
    with torch.no_grad():
        image_features = model.encode_image(images[:actual_samples])
        logit_scale = model.logit_scale.exp()
        similarity = logit_scale * image_features @ text_features.t()
        probs = similarity.softmax(dim=-1)

    fig, axes = plt.subplots(actual_samples, 2, figsize=(12, 3 * actual_samples))

    for i in range(actual_samples):
        top3_probs, top3_indices = probs[i].topk(3)

        # 显示图像
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        ax_img = axes[i, 0] if actual_samples > 1 else axes[0]
        ax_bar = axes[i, 1] if actual_samples > 1 else axes[1]

        ax_img.imshow(img)
        # 显示 True Label (需防止越界)
        true_label_idx = labels[i].item()
        true_name = class_names[true_label_idx] if true_label_idx < len(class_names) else f"ID {true_label_idx}"
        ax_img.set_title(f"True: {true_name}")
        ax_img.axis('off')

        # 显示预测概率
        ax_bar.barh(range(3), top3_probs.cpu().numpy())
        ax_bar.set_yticks(range(3))
        # 转换预测 ID 为名字
        pred_names = [class_names[idx] if idx < len(class_names) else str(idx) for idx in top3_indices.cpu().numpy()]
        ax_bar.set_yticklabels(pred_names)
        ax_bar.set_xlabel('Probability')
        ax_bar.set_title('Top-3 Predictions')
        ax_bar.set_xlim([0, 1])

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
    parser.add_argument('--data_root', type=str, default='./data/ImageNet-Custom', help='ImageNet data root')
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots per class')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--adapter_dim', type=int, default=64, help='Adapter dimension')
    parser.add_argument('--alpha', type=float, default=0.2, help='Adapter residual weight')
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/32', help='CLIP backbone')
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

    # [修改] 验证集也使用 ImageFolder
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess
    ])

    val_path = os.path.join(args.data_root, 'val')
    if not os.path.exists(val_path):
        # 尝试使用根目录（如果数据集没分split）
        val_path = args.data_root

    full_val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

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

    # [修改] 获取类别名称 (ID -> Name 映射)
    print("Mapping Class IDs to English Names...")
    raw_class_ids = train_dataset.classes  # 这些是 nxxxxxx

    # 尝试加载 json
    json_path = os.path.join(args.data_root, "imagenet_class_index.json")
    if not os.path.exists(json_path):
        # 尝试在当前目录找
        json_path = "imagenet_class_index.json"

    class_names = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
        id_to_name_map = {v[0]: v[1] for k, v in raw_json.items()}

        for folder_id in raw_class_ids:
            # 获取名字，替换下划线
            name = id_to_name_map.get(folder_id, folder_id).replace('_', ' ')
            class_names.append(name)
    else:
        print("Warning: JSON not found, using folder IDs as class names.")
        class_names = raw_class_ids

    # 4. 准备文本提示
    print("Preparing text prompts...")
    text_prompts = prepare_text_prompts(class_names)

    # 5. 评估原始CLIP的zero-shot性能
    print("\n" + "=" * 50)
    print("Evaluating original CLIP (zero-shot)...")
    print("=" * 50)

    original_clip_model = clip_model
    original_text_features = original_clip_model.encode_text(text_prompts.to(args.device))
    # [修改] 转 float
    original_text_features = original_text_features.float()
    original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            image_features = original_clip_model.encode_image(images)
            # [修改] 转 float
            image_features = image_features.float()
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
    if original_acc > 0:
        print(f"Relative improvement: {100 * (final_acc - original_acc) / original_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()