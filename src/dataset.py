import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CutPaste(A.ImageOnlyTransform):
    """
    Phép biến đổi CutPaste (Google Research, CVPR 2021):
    Cắt một vùng patch ngẫu nhiên từ ảnh và dán vào vị trí khác để mô phỏng khuyết tật bề mặt sản phẩm.
    """
    def __init__(
        self,
        area_ratio: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio: Tuple[float, float] = (0.3, 3.3),
        always_apply: bool = False,
        p: float = 0.5
    ):
        super().__init__(p=p)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w, c = img.shape
        img_area = h * w

        # Tính diện tích và kích thước patch
        patch_area = np.random.uniform(self.area_ratio[0], self.area_ratio[1]) * img_area
        aspect = np.exp(np.random.uniform(np.log(self.aspect_ratio[0]), np.log(self.aspect_ratio[1])))

        patch_w = int(np.clip(np.round(np.sqrt(patch_area * aspect)), 4, w - 1))
        patch_h = int(np.clip(np.round(np.sqrt(patch_area / aspect)), 4, h - 1))

        if patch_w >= w or patch_h >= h:
            return img

        # Tọa độ cắt
        from_x = np.random.randint(0, w - patch_w)
        from_y = np.random.randint(0, h - patch_h)
        patch = img[from_y : from_y + patch_h, from_x : from_x + patch_w].copy()

        # Biến đổi nhẹ patch (rotation hoặc đổi độ sáng)
        if np.random.rand() > 0.5:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
            patch_h, patch_w = patch.shape[:2]

        if patch_w >= w or patch_h >= h:
            return img

        # Tọa độ dán
        to_x = np.random.randint(0, w - patch_w)
        to_y = np.random.randint(0, h - patch_h)

        out_img = img.copy()
        out_img[to_y : to_y + patch_h, to_x : to_x + patch_w] = patch
        return out_img


def get_transforms(
    img_size: Tuple[int, int] = (224, 224), 
    is_train: bool = True,
    use_cutpaste: bool = False
) -> A.Compose:
    """
    Xây dựng pipeline augmentation chuẩn cho defect detection trong môi trường công nghiệp:
    - CLAHE: Cân bằng độ tương phản bề mặt vật liệu.
    - CutPaste: Mô phỏng khuyết tật công nghiệp (vết ghép/dị vật).
    - ColorJitter & Flips/Rotations: Tăng cường tính bất biến theo ánh sáng và góc chụp.
    - ImageNet Normalization.
    """
    if is_train:
        transforms_list = [
            A.Resize(img_size[0], img_size[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)
        ]
        if use_cutpaste:
            transforms_list.append(CutPaste(p=0.5))

        transforms_list.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return A.Compose(transforms_list)

    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class DefectDataset(Dataset):
    """
    Dataset nhị phân cho bài toán Defect Detection:
    - 0: Normal / Good
    - 1: Defect / Anomaly
    """
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[A.Compose] = None
    ):
        if len(image_paths) != len(labels):
            raise ValueError(f"Số lượng paths ({len(image_paths)}) và labels ({len(labels)}) phải bằng nhau.")
        
        self.image_paths = [str(p) for p in image_paths]
        self.labels = [int(l) for l in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Không thể đọc file hoặc ảnh bị hỏng tại đường dẫn: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor

    @classmethod
    def from_directory(cls, split_dir: str, transform: Optional[A.Compose] = None) -> "DefectDataset":
        """
        Nạp dữ liệu từ cấu trúc thư mục:
        split_dir/
          ├── good/
          └── defect/
        """
        path = Path(split_dir)
        if not path.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {split_dir}")

        image_paths = []
        labels = []

        # Class 0: Good
        good_dir = path / "good"
        if good_dir.exists():
            for f in good_dir.glob("*.*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(str(f))
                    labels.append(0)

        # Class 1: Defect
        defect_dir = path / "defect"
        if defect_dir.exists():
            for f in defect_dir.glob("*.*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(str(f))
                    labels.append(1)

        return cls(image_paths=image_paths, labels=labels, transform=transform)


def get_class_imbalance_info(labels: List[int]) -> Dict[str, float]:
    """Tính toán tỷ lệ phân bố lớp và tỷ lệ mất cân bằng (Imbalance Ratio)."""
    labels_arr = np.array(labels)
    num_good = int(np.sum(labels_arr == 0))
    num_defect = int(np.sum(labels_arr == 1))
    total = len(labels_arr)
    ratio = (num_good / num_defect) if num_defect > 0 else float("inf")
    return {
        "num_good": num_good,
        "num_defect": num_defect,
        "total": total,
        "imbalance_ratio": ratio,
        "defect_percentage": (num_defect / total * 100) if total > 0 else 0.0
    }


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Tạo WeightedRandomSampler để cân bằng xác suất lấy mẫu cho từng class."""
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=2)
    
    # Tránh chia cho 0
    class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
    sample_weights = [class_weights[l] for l in labels_arr]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def create_dataloaders(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khởi tạo DataLoaders cho Train, Val, Test từ thư mục processed data.
    Tự động áp dụng WeightedRandomSampler cho tập Train nếu được cấu hình.
    """
    train_transform = get_transforms(img_size=img_size, is_train=True)
    eval_transform = get_transforms(img_size=img_size, is_train=False)

    train_ds = DefectDataset.from_directory(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = DefectDataset.from_directory(os.path.join(data_dir, "val"), transform=eval_transform)
    test_ds = DefectDataset.from_directory(os.path.join(data_dir, "test"), transform=eval_transform)

    train_info = get_class_imbalance_info(train_ds.labels)
    pin_memory = torch.cuda.is_available()

    if use_weighted_sampler and train_info['num_defect'] > 0:
        sampler = create_weighted_sampler(train_ds.labels)
        
        # Báo cáo tỷ lệ trước và sau sampler
        sampled_indices = list(sampler)
        sampled_labels = [train_ds.labels[i] for i in sampled_indices]
        sampled_defect_pct = (sum(sampled_labels) / len(sampled_labels)) * 100
        
        print("\n" + "-"*55)
        print("[DataLoader] BÁO CÁO MẤT CÂN BẰNG DỮ LIỆU:")
        print(f"  - Trước Sampler: Good={train_info['num_good']}, Defect={train_info['num_defect']} "
              f"({train_info['defect_percentage']:.1f}% Defect | Tỷ lệ {train_info['imbalance_ratio']:.1f}:1)")
        print(f"  - Sau Sampler  : ~{sampled_defect_pct:.1f}% Defect (Đã cân bằng ~50:50)")
        print("-" * 55 + "\n")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        print(f"[DataLoader] Không dùng WeightedRandomSampler: Defect={train_info['defect_percentage']:.1f}%")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
