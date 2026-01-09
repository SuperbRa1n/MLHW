import os
# ========================================================
# 关键修复：设置 HF 镜像站 (必须在 import datasets 之前设置)
# 这能解决 [Errno 101] Network is unreachable 问题
# ========================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import shutil
import random
import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 1. 每个类别总共下载多少张？ (建议 200 或 500)
# 比如 200 张：最终可能会划分为 160张训练 + 40张验证
TOTAL_IMAGES_PER_CLASS = 32

# 2. 验证集占比 (0.2 表示 20% 做验证集)
VAL_RATIO = 0.5

# 3. 你的 HF Token (必须配置，因为 ImageNet 是受限数据)
# 去 https://huggingface.co/settings/tokens 申请
# 如果你不想用命令行登录，可以在这里填入你的 HF Token
HF_TOKEN = ""
# HF_TOKEN = True  # True 表示使用本地 huggingface-cli login 的缓存


# 4. 数据保存路径
ROOT_DIR = "./data/ImageNet-Custom"
RAW_DIR = os.path.join(ROOT_DIR, "raw_pool")  # 临时下载池
FINAL_TRAIN_DIR = os.path.join(ROOT_DIR, "train")
FINAL_VAL_DIR = os.path.join(ROOT_DIR, "val")
JSON_PATH = os.path.join(ROOT_DIR, "imagenet_class_index.json")


# ==============================================

def download_json_mapping():
    """下载类别映射文件"""
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    print(f"1. [准备] 下载类别映射文件...")

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    try:
        r = requests.get(url, timeout=20)
        with open(JSON_PATH, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"   ❌ JSON 下载失败: {e}")
        return False


def get_class_id_map():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    idx_to_folder = {}
    for idx_str, (folder_id, class_name) in data.items():
        idx_to_folder[int(idx_str)] = folder_id
    return idx_to_folder


def step1_download_to_pool(idx_to_folder):
    """步骤1: 将所有图片下载到一个总池子里"""
    print(f"\n2. [下载] 开始流式下载数据 (每类 {TOTAL_IMAGES_PER_CLASS} 张)...")

    # 计数器
    counts = {i: 0 for i in range(1000)}
    total_needed = 1000 * TOTAL_IMAGES_PER_CLASS

    try:
        # 只从 'train' split 下载，因为那里数据最全
        dataset = load_dataset(
            "imagenet-1k",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"\n❌ HF 连接失败: {e}")
        print("请检查 Token 是否正确，且是否已在官网同意 ImageNet 协议。")
        return False

    pbar = tqdm(total=total_needed, unit="img")

    for item in dataset:
        label_idx = item['label']

        # 如果该类已满，跳过
        if counts[label_idx] >= TOTAL_IMAGES_PER_CLASS:
            continue

        folder_name = idx_to_folder[label_idx]

        # 保存到 raw_pool/nxxxxxx/
        class_dir = os.path.join(RAW_DIR, folder_name)
        os.makedirs(class_dir, exist_ok=True)

        try:
            image = item['image']
            if image.mode != "RGB":
                image = image.convert("RGB")

            filename = f"{folder_name}_{counts[label_idx]}.jpg"
            save_path = os.path.join(class_dir, filename)
            image.save(save_path, "JPEG", quality=90)

            counts[label_idx] += 1
            pbar.update(1)
        except:
            pass

        # 检查是否全部完成
        # 性能优化：每 100 张检查一次全局状态
        if pbar.n % 100 == 0:
            if all(c >= TOTAL_IMAGES_PER_CLASS for c in counts.values()):
                break

    pbar.close()
    print("✅ 下载阶段完成！")
    return True


def step2_split_dataset():
    """步骤2: 本地划分训练集和验证集"""
    print(f"\n3. [划分] 正在将数据划分为 Train/Val (比例 {VAL_RATIO})...")

    if not os.path.exists(RAW_DIR):
        print("❌ 未找到下载的数据池！")
        return

    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

    for class_name in tqdm(classes, desc="Processing Classes"):
        src_class_dir = os.path.join(RAW_DIR, class_name)

        # 获取该类所有图片
        images = os.listdir(src_class_dir)
        random.shuffle(images)  # 打乱顺序

        # 计算切分点
        num_val = int(len(images) * VAL_RATIO)
        val_images = images[:num_val]
        train_images = images[num_val:]

        # 移动到 Train 目录
        dst_train_dir = os.path.join(FINAL_TRAIN_DIR, class_name)
        os.makedirs(dst_train_dir, exist_ok=True)
        for img in train_images:
            shutil.move(os.path.join(src_class_dir, img), os.path.join(dst_train_dir, img))

        # 移动到 Val 目录
        dst_val_dir = os.path.join(FINAL_VAL_DIR, class_name)
        os.makedirs(dst_val_dir, exist_ok=True)
        for img in val_images:
            shutil.move(os.path.join(src_class_dir, img), os.path.join(dst_val_dir, img))

    # 清理空目录
    try:
        shutil.rmtree(RAW_DIR)
        print("   -> 已清理临时文件。")
    except:
        pass

    print("✅ 划分完成！")


def main():
    print("=" * 50)
    print("  ImageNet 全自动下载与划分脚本 (One-Stop)")
    print("=" * 50)

    if "你的Token" in HF_TOKEN:
        print("❌ 请先编辑脚本，填入你的 Hugging Face Token！")
        return

    # 1. 准备 JSON
    if not download_json_mapping():
        return

    idx_to_folder = get_class_id_map()

    # 2. 下载所有数据到池子
    if step1_download_to_pool(idx_to_folder):
        # 3. 执行划分
        step2_split_dataset()

        print("\n" + "=" * 50)
        print("🎉 全部任务执行完毕！")
        print(f"数据根目录: {os.path.abspath(ROOT_DIR)}")
        print(f"  - 训练集: {os.path.abspath(FINAL_TRAIN_DIR)}")
        print(f"  - 验证集: {os.path.abspath(FINAL_VAL_DIR)}")
        print(f"  - 映射表: {os.path.abspath(JSON_PATH)}")
        print("\n现在 Train 和 Val 的文件夹结构是 100% 对齐的，你可以放心训练了！")
        print("=" * 50)


if __name__ == "__main__":
    # 设置随机种子，保证每次划分结果一致 (如果数据源不变)
    random.seed(42)
    main()