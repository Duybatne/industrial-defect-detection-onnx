import os
import sys
import shutil
import random
import argparse
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def generate_synthetic_industrial_data(raw_dir: str, category: str = "bottle", num_good_train: int = 150, num_defect: int = 50, seed: int = 42):
    """
    Tạo dữ liệu giả lập chuẩn môi trường công nghiệp cho kiểm thử và huấn luyện baseline.
    Mô phỏng bề mặt vật liệu và các loại khuyết tật: vết xước (scratch), nứt (crack), nhiễm bẩn (contamination), lỗ hổng (hole).
    """
    np.random.seed(seed)
    random.seed(seed)

    raw_path = Path(raw_dir)
    train_good_dir = raw_path / "train" / "good"
    test_good_dir = raw_path / "test" / "good"
    defect_types = ["broken_large", "contamination", "crack", "scratch"]
    
    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)
    for dt in defect_types:
        (raw_path / "test" / dt).mkdir(parents=True, exist_ok=True)

    def create_base_surface(width=512, height=512) -> np.ndarray:
        # Giả lập bề mặt kim loại / thủy tinh / vật liệu công nghiệp với gradient và texture
        base_color = np.random.randint(180, 220)
        img = np.full((height, width, 3), base_color, dtype=np.uint8)
        
        # Thêm vân texture nhẹ
        noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Thêm vùng sản phẩm (ví dụ thân chai / linh kiện hình trụ hoặc chữ nhật bo góc)
        center_x, center_y = width // 2, height // 2
        cv2.ellipse(img, (center_x, center_y), (width // 3, height // 2 - 20), 0, 0, 360, (230, 235, 240), -1)
        cv2.ellipse(img, (center_x, center_y), (width // 3, height // 2 - 20), 0, 0, 360, (140, 150, 160), 3)
        return img

    def add_defect(img: np.ndarray, defect_type: str) -> np.ndarray:
        h, w, _ = img.shape
        out = img.copy()
        cx, cy = np.random.randint(w // 3, 2 * w // 3), np.random.randint(h // 4, 3 * h // 4)

        if defect_type == "scratch":
            # Vẽ các đường rạch ngẫu nhiên
            pts = np.array([
                [cx, cy],
                [cx + np.random.randint(-40, 40), cy + np.random.randint(20, 80)],
                [cx + np.random.randint(-50, 50), cy + np.random.randint(80, 140)]
            ], np.int32)
            cv2.polylines(out, [pts], False, (40, 40, 50), thickness=np.random.randint(2, 4))

        elif defect_type == "contamination":
            # Đốm bẩn / vệt dầu mỡ
            for _ in range(np.random.randint(3, 8)):
                rx = cx + np.random.randint(-30, 30)
                ry = cy + np.random.randint(-30, 30)
                cv2.circle(out, (rx, ry), np.random.randint(5, 18), (30, 35, 40), -1)

        elif defect_type == "crack":
            # Vết nứt phân nhánh
            cv2.line(out, (cx, cy), (cx + 35, cy + 45), (20, 20, 20), 2)
            cv2.line(out, (cx + 35, cy + 45), (cx + 60, cy + 85), (20, 20, 20), 2)
            cv2.line(out, (cx + 35, cy + 45), (cx + 20, cy + 90), (20, 20, 20), 1)

        elif defect_type == "broken_large":
            # Vỡ mảng lớn
            cv2.ellipse(out, (cx, cy), (np.random.randint(30, 60), np.random.randint(20, 45)), np.random.randint(0, 180), 0, 360, (20, 25, 30), -1)

        return out

    print(f"[+] Đang sinh {num_good_train} ảnh Good cho tập train...")
    for i in range(num_good_train):
        img = create_base_surface()
        cv2.imwrite(str(train_good_dir / f"{category}_good_train_{i:04d}.png"), img)

    print(f"[+] Đang sinh 30 ảnh Good cho tập test...")
    for i in range(30):
        img = create_base_surface()
        cv2.imwrite(str(test_good_dir / f"{category}_good_test_{i:04d}.png"), img)

    print(f"[+] Đang sinh các loại lỗi ({defect_types}) cho tập test...")
    samples_per_type = max(1, num_defect // len(defect_types))
    for dt in defect_types:
        dt_dir = raw_path / "test" / dt
        for i in range(samples_per_type):
            img = create_base_surface()
            defective_img = add_defect(img, dt)
            cv2.imwrite(str(dt_dir / f"{category}_{dt}_{i:04d}.png"), defective_img)

    print(f"[✓] Đã tạo thành công dataset synthetic tại {raw_dir}")


def download_mvtec_dataset(category: str, raw_dir: str) -> bool:
    """Tải dataset MVTec AD thực tế cho category được chỉ định."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    url = f"https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951964/{category}.tar.xz"
    tar_path = raw_path / f"{category}.tar.xz"
    
    print(f"[+] Đang tải MVTec AD ({category}) từ {url}...")
    try:
        urllib.request.urlretrieve(url, str(tar_path))
        print(f"[+] Đang giải nén {tar_path}...")
        with tarfile.open(str(tar_path)) as tar:
            tar.extractall(path=raw_path)
        tar_path.unlink(missing_ok=True)
        print(f"[✓] Đã tải và giải nén thành công MVTec AD ({category})")
        return True
    except Exception as e:
        print(f"[!] Không thể tải MVTec AD trực tuyến ({e}). Chuyển sang chế độ synthetic generator.")
        return False


def process_and_split_dataset(raw_dir: str, processed_dir: str, val_ratio: float = 0.1, test_ratio: float = 0.15, seed: int = 42) -> Dict[str, Dict[str, int]]:
    """
    Đọc dữ liệu từ thư mục raw, gán nhãn nhị phân (good=0, defect=1), 
    thực hiện Stratified Split và copy vào processed/{train, val, test}/{good, defect}.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    if processed_path.exists():
        shutil.rmtree(processed_path)
        
    for split in ["train", "val", "test"]:
        (processed_path / split / "good").mkdir(parents=True, exist_ok=True)
        (processed_path / split / "defect").mkdir(parents=True, exist_ok=True)

    # Thu thập tất cả các file từ raw
    image_entries: List[Tuple[Path, int]] = [] # (path, label: 0=good, 1=defect)

    # raw/train/good
    for f in (raw_path / "train" / "good").glob("*.*"):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            image_entries.append((f, 0))

    # raw/test/*
    test_dir = raw_path / "test"
    if test_dir.exists():
        for category_dir in test_dir.iterdir():
            if category_dir.is_dir():
                label = 0 if category_dir.name.lower() == "good" else 1
                for f in category_dir.glob("*.*"):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                        image_entries.append((f, label))

    if not image_entries:
        raise ValueError(f"Không tìm thấy ảnh nào trong thư mục {raw_dir}")

    paths = [item[0] for item in image_entries]
    labels = [item[1] for item in image_entries]

    # Stratified split: Train, Val, Test
    # Bước 1: Tách test_ratio
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )

    # Bước 2: Tách val_ratio từ train_val
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=adjusted_val_ratio, random_state=seed, stratify=train_val_labels
    )

    split_data = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }

    summary: Dict[str, Dict[str, int]] = {}

    for split_name, (sp_paths, sp_labels) in split_data.items():
        summary[split_name] = {"good": 0, "defect": 0}
        for src_path, lbl in zip(sp_paths, sp_labels):
            label_name = "good" if lbl == 0 else "defect"
            dst_dir = processed_path / split_name / label_name
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            summary[split_name][label_name] += 1

    print("\n" + "="*50)
    print("THỐNG KÊ PHÂN CHIA DỮ LIỆU (STRATIFIED SPLIT)")
    print("="*50)
    for split_name, counts in summary.items():
        total = counts["good"] + counts["defect"]
        defect_pct = (counts["defect"] / total * 100) if total > 0 else 0
        print(f"[{split_name.upper():<5}] Tổng: {total:<4} | Good: {counts['good']:<4} | Defect: {counts['defect']:<4} ({defect_pct:.1f}%)")
    print("="*50 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Tải, sinh và phân chia tập dữ liệu Defect Detection")
    parser.add_argument("--category", type=str, default="bottle", help="Category dataset MVTec AD")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Thư mục chứa raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Thư mục chứa processed data")
    parser.add_argument("--synthetic", action="store_true", help="Ưu tiên tạo tập dữ liệu synthetic")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỷ lệ tập validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Tỷ lệ tập test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    # Kiểm tra nếu raw chưa có ảnh hoặc yêu cầu synthetic
    has_raw_data = raw_path.exists() and any(raw_path.glob("**/*.png"))

    if args.synthetic or not has_raw_data:
        if not args.synthetic:
            success = download_mvtec_dataset(args.category, args.raw_dir)
            if not success:
                generate_synthetic_industrial_data(args.raw_dir, category=args.category, seed=args.seed)
        else:
            generate_synthetic_industrial_data(args.raw_dir, category=args.category, seed=args.seed)

    process_and_split_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
