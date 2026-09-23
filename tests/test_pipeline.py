import os
import pytest
import numpy as np
import torch
import yaml
from src.models.network import build_model
from src.utils.metrics import calculate_metrics


def test_model_forward():
    config = {
        "model": {
            "backbone": "resnet18",
            "num_classes": 2,
            "pretrained": False
        }
    }
    model = build_model(config)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 2)


def test_metrics_calculation():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == 0.8
