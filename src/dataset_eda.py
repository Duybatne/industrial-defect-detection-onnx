import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import DefectDataset, get_transforms, get_class_imbalance_info


def analyze_dataset(data_dir: str) -> Dict[str, dict]:
    """Phân tích số lượng mẫu, kích thước và phân phối ánh sáng của ảnh."""
    data_path = Path(data_dir)
    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            continue

        good_files = list((split_dir / "good").glob("*.*"))
        defect_files = list((split_dir / "defect").glob("*.*"))
        
        all_files = [(f, 0) for f in good_files] + [(f, 1) for f in defect_files]
        labels = [item[1] for item in all_files]
        imbalance_info = get_class_imbalance_info(labels)

        # Tính toán phân phối kích thước và độ sáng
        brightness_list = []
        contrast_list = []
        dimensions = []

        for f, _ in all_files[:50]: # Sample 50 ảnh để thống kê nhanh
            img = cv2.imread(str(f))
            if img is not None:
                h, w, c = img.shape
                dimensions.append((w, h))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_list.append(np.mean(gray))
                contrast_list.append(np.std(gray))

        stats[split] = {
            "imbalance_info": imbalance_info,
            "dimensions": set(dimensions),
            "mean_brightness": float(np.mean(brightness_list)) if brightness_list else 0.0,
            "mean_contrast": float(np.mean(contrast_list)) if contrast_list else 0.0,
            "good_files": [str(f) for f in good_files],
            "defect_files": [str(f) for f in defect_files]
        }

    return stats


def plot_eda_distribution(stats: Dict[str, dict], output_path: str):
    """Vẽ biểu đồ phân bố lớp và tỷ lệ mất cân bằng dữ liệu."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    splits = list(stats.keys())
    good_counts = [stats[s]["imbalance_info"]["num_good"] for s in splits]
    defect_counts = [stats[s]["imbalance_info"]["num_defect"] for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart so sánh Good vs Defect trên từng tập
    ax1.bar(x - width/2, good_counts, width, label="Normal / Good (0)", color="#2b5c8f")
    ax1.bar(x + width/2, defect_counts, width, label="Defect / Anomaly (1)", color="#d9534f")
    ax1.set_xlabel("Dataset Split", fontweight="bold")
    ax1.set_ylabel("Số lượng mẫu", fontweight="bold")
    ax1.set_title("Phân bố số lượng mẫu theo tập (Train/Val/Test)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in splits])
    ax1.legend()

    # Pie chart tổng thể toàn bộ dataset
    total_good = sum(good_counts)
    total_defect = sum(defect_counts)
    ax2.pie(
        [total_good, total_defect], 
        labels=[f"Good ({total_good})", f"Defect ({total_defect})"],
        autopct="%1.1f%%",
        colors=["#2b5c8f", "#d9534f"],
        startangle=140,
        explode=(0, 0.1),
        shadow=True
    )
    ax2.set_title(f"Tỷ lệ Imbalance Tổng thể (Tỷ lệ {total_good/max(1, total_defect):.1f}:1)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ phân bố tại: {output_path}")


def plot_eda_samples(stats: Dict[str, dict], output_path: str):
    """Vẽ ma trận ảnh mẫu (Good vs Defect) cùng hiệu ứng sau tiền xử lý Augmentation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    train_good = stats.get("train", {}).get("good_files", [])
    train_defect = stats.get("train", {}).get("defect_files", [])

    if not train_good or not train_defect:
        print("[!] Không đủ ảnh để vẽ ma trận mẫu.")
        return

    transform = get_transforms(img_size=(224, 224), is_train=True)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # Hàng 1: 4 ảnh Good gốc
    for i in range(4):
        img_p = train_good[i % len(train_good)]
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title("Good (Original)", color="navy", fontsize=10)
        axes[0, i].axis("off")

    # Hàng 2: 4 ảnh Defect gốc
    for i in range(4):
        img_p = train_defect[i % len(train_defect)]
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title("Defect (Original)", color="darkred", fontsize=10)
        axes[1, i].axis("off")

    # Hàng 3: 4 ảnh Good sau Augmentation (CLAHE + Jitter)
    for i in range(4):
        img_p = train_good[i % len(train_good)]
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        aug = transform(image=img)["image"].permute(1, 2, 0).numpy()
        # De-normalize để hiển thị
        aug = np.clip((aug * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)), 0, 1)
        axes[2, i].imshow(aug)
        axes[2, i].set_title("Good (Augmented)", color="navy", fontsize=10)
        axes[2, i].axis("off")

    # Hàng 4: 4 ảnh Defect sau Augmentation (CLAHE + Jitter)
    for i in range(4):
        img_p = train_defect[i % len(train_defect)]
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        aug = transform(image=img)["image"].permute(1, 2, 0).numpy()
        aug = np.clip((aug * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)), 0, 1)
        axes[3, i].imshow(aug)
        axes[3, i].set_title("Defect (Augmented)", color="darkred", fontsize=10)
        axes[3, i].axis("off")

    plt.suptitle("Trực Quan Hóa Mẫu Ảnh Gốc & Sau Pipeline Augmentation (CLAHE, Flips, Jitter)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu ma trận ảnh mẫu tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Khám phá và phân tích chất lượng dữ liệu (EDA)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Thư mục dữ liệu đã xử lý")
    parser.add_argument("--output_dir", type=str, default="reports", help="Thư mục lưu báo cáo biểu đồ")
    args = parser.parse_args()

    print(f"[+] Bắt đầu phân tích EDA trên dữ liệu tại: {args.data_dir}")
    stats = analyze_dataset(args.data_dir)

    print("\n" + "="*60)
    print(f"{'TẬP DỮ LIỆU':<10} | {'GOOD':<8} | {'DEFECT':<8} | {'TỶ LỆ DEFECT':<14} | {'ĐỘ SÁNG TB':<10}")
    print("="*60)
    for split, info in stats.items():
        imb = info["imbalance_info"]
        print(f"{split.upper():<10} | {imb['num_good']:<8} | {imb['num_defect']:<8} | {imb['defect_percentage']:<13.1f}% | {info['mean_brightness']:<10.1f}")
    print("="*60 + "\n")

    dist_path = os.path.join(args.output_dir, "eda_distribution.png")
    samples_path = os.path.join(args.output_dir, "eda_samples.png")

    plot_eda_distribution(stats, dist_path)
    plot_eda_samples(stats, samples_path)


if __name__ == "__main__":
    main()
