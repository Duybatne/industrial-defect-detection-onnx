import os
import shutil
import tempfile
import pytest
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DefectDataset,
    get_transforms,
    get_class_imbalance_info,
    create_weighted_sampler,
    create_dataloaders
)


@pytest.fixture
def dummy_dataset_dir():
    """Tạo một thư mục dữ liệu giả lập tạm thời cho kiểm thử."""
    temp_dir = tempfile.mkdtemp()
    
    for split in ["train", "val", "test"]:
        good_dir = os.path.join(temp_dir, split, "good")
        defect_dir = os.path.join(temp_dir, split, "defect")
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(defect_dir, exist_ok=True)

        # Tạo ảnh giả lập (Imbalanced: 20 good, 4 defect)
        for i in range(20):
            img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(good_dir, f"good_{i}.png"), img)
            
        for i in range(4):
            img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(defect_dir, f"defect_{i}.png"), img)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_transforms_output_shape_and_range():
    """Kiểm tra pipeline transforms trả về đúng tensor [3, 224, 224] không chứa NaN/Inf."""
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    
    train_transform = get_transforms(img_size=(224, 224), is_train=True)
    val_transform = get_transforms(img_size=(224, 224), is_train=False)

    train_out = train_transform(image=dummy_img)["image"]
    val_out = val_transform(image=dummy_img)["image"]

    assert isinstance(train_out, torch.Tensor)
    assert train_out.shape == (3, 224, 224)
    assert not torch.isnan(train_out).any(), "Transform sinh ra giá trị NaN"
    assert not torch.isinf(train_out).any(), "Transform sinh ra giá trị Inf"

    assert isinstance(val_out, torch.Tensor)
    assert val_out.shape == (3, 224, 224)
    assert not torch.isnan(val_out).any()
    assert not torch.isinf(val_out).any()


def test_defect_dataset_binary_labels(dummy_dataset_dir):
    """Kiểm tra Dataset gán nhãn nhị phân chuẩn (0: Good, 1: Defect) kiểu torch.long."""
    train_dir = os.path.join(dummy_dataset_dir, "train")
    transform = get_transforms(img_size=(224, 224), is_train=False)
    dataset = DefectDataset.from_directory(train_dir, transform=transform)

    assert len(dataset) == 24 # 20 good + 4 defect

    unique_labels = set()
    for i in range(len(dataset)):
        img_tensor, label_tensor = dataset[i]
        assert img_tensor.shape == (3, 224, 224)
        assert label_tensor.dtype == torch.long
        unique_labels.add(label_tensor.item())

    assert unique_labels == {0, 1}


def test_defect_dataset_invalid_image():
    """Kiểm tra xử lý ngoại lệ an toàn khi nạp file ảnh không tồn tại hoặc lỗi."""
    dataset = DefectDataset(image_paths=["/path/to/non_existent_file.png"], labels=[0])
    with pytest.raises(ValueError, match="Không thể đọc file hoặc ảnh bị hỏng"):
        _ = dataset[0]


def test_weighted_random_sampler():
    """Kiểm tra WeightedRandomSampler cân bằng xác suất lấy mẫu khi dữ liệu bị mất cân bằng nặng."""
    # 90 mẫu good (0) và 10 mẫu defect (1) -> tỷ lệ 9:1
    labels = [0] * 90 + [1] * 10
    sampler = create_weighted_sampler(labels)

    # Lấy mẫu 1000 lần thông qua sampler
    sampled_indices = list(sampler)
    sampled_labels = [labels[idx] for idx in sampled_indices]
    
    defect_count = sum(sampled_labels)
    defect_ratio = defect_count / len(sampled_labels)

    # Với WeightedRandomSampler lý tưởng là ~50% defect (dung sai cho phép 35% - 65%)
    assert 0.35 <= defect_ratio <= 0.65, f"Tỷ lệ defect sau sampler: {defect_ratio:.2f} không đạt độ cân bằng"


def test_create_dataloaders(dummy_dataset_dir):
    """Kiểm tra khởi tạo DataLoaders hoàn chỉnh từ processed directory."""
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=dummy_dataset_dir,
        img_size=(224, 224),
        batch_size=8,
        num_workers=0,
        use_weighted_sampler=True
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (8, 3, 224, 224)
    assert batch_y.shape == (8,)
    assert batch_y.dtype == torch.long


def test_cutpaste_augmentation():
    """Kiểm tra CutPaste transform hoạt động và trả về ảnh đúng kích thước."""
    from src.dataset import CutPaste
    dummy_img = np.ones((256, 256, 3), dtype=np.uint8) * 100
    transform = CutPaste(p=1.0)
    augmented = transform(image=dummy_img)["image"]

    assert augmented.shape == dummy_img.shape
    assert augmented.dtype == dummy_img.dtype

