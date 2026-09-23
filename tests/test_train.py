import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.network import build_model, DefectClassifier
from src.models.losses import FocalLoss, build_loss_fn
from src.utils.metrics import calculate_metrics, find_optimal_threshold


@pytest.fixture
def dummy_config():
    return {
        "model": {
            "backbone": "resnet18",
            "num_classes": 2,
            "pretrained": False,
            "in_channels": 3,
            "drop_rate": 0.2
        },
        "training": {
            "loss_type": "focal",
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
            "lr": 0.001
        }
    }


def test_model_forward_backward(dummy_config):
    """Kiểm tra mô hình forward và backward qua 1 batch không bị lỗi gradient."""
    model = build_model(dummy_config)
    model.train()
    
    dummy_input = torch.randn(4, 3, 224, 224, requires_grad=True)
    dummy_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    outputs = model(dummy_input)
    assert outputs.shape == (4, 2)

    loss_fn = FocalLoss()
    loss = loss_fn(outputs, dummy_targets)
    loss.backward()

    # Kiểm tra gradient đã được tính
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient của {name} bị None"


def test_focal_loss_computation():
    """Kiểm tra Focal Loss tính toán chính xác, không sinh NaN/Inf."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    
    logits = torch.tensor([[2.0, -1.0], [-1.5, 3.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0


def test_model_freeze_unfreeze(dummy_config):
    """Kiểm tra logic đóng băng (freeze) và mở khóa (unfreeze) backbone."""
    model = build_model(dummy_config)
    
    # Freeze
    model.freeze_backbone()
    for param in model.model.parameters():
        assert not param.requires_grad, "Backbone parameter vẫn còn requires_grad=True khi bị freeze"
    for param in model.classifier.parameters():
        assert param.requires_grad, "Classifier head parameter phải requires_grad=True"

    # Unfreeze
    model.unfreeze_backbone()
    for param in model.parameters():
        assert param.requires_grad, "Tất cả parameters phải requires_grad=True sau unfreeze"


def test_overfit_single_batch(dummy_config):
    """Kiểm tra mô hình học và giảm loss liên tục trên 1 batch nhỏ (Overfit Single Batch test)."""
    model = build_model(dummy_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = loss_fn(out, dummy_targets)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.3, (
        f"Mô hình không học được trên single batch: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_checkpoint_save_and_load(dummy_config):
    """Kiểm tra lưu và nạp checkpoint chính xác 100% trọng số."""
    model1 = build_model(dummy_config)
    model1.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    out1 = model1(dummy_input)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({"model_state_dict": model1.state_dict(), "val_f1": 0.95}, tmp_path)

        model2 = build_model(dummy_config)
        checkpoint = torch.load(tmp_path, weights_only=False)
        model2.load_state_dict(checkpoint["model_state_dict"])
        model2.eval()

        out2 = model2(dummy_input)
        assert torch.allclose(out1, out2, atol=1e-5), "Output của model sau khi load checkpoint không khớp!"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_find_optimal_threshold():
    """Kiểm tra thuật toán tìm ngưỡng T* đạt Recall mục tiêu >= 0.98."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # Xác suất dự đoán có sự không chắc chắn ở ngưỡng 0.5
    y_prob = np.array([0.05, 0.15, 0.20, 0.45, 0.48, 0.55, 0.70, 0.85, 0.90, 0.95])

    opt_t, metrics = find_optimal_threshold(y_true, y_prob, target_recall=0.98)
    assert metrics["recall"] >= 0.98
    assert opt_t <= 0.48, f"Ngưỡng tối ưu phải bao quát mẫu defect có prob 0.48, nhận được: {opt_t}"
