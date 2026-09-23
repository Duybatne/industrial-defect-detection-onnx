import os
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models.network import build_model
from src.pipeline.inference import ONNXInferencer


@pytest.fixture(scope="module")
def onnx_model_path():
    path = "weights/model.onnx"
    if not os.path.exists(path):
        pytest.skip(f"ONNX model file {path} not found.")
    return path


@pytest.fixture(scope="module")
def quantized_model_path():
    path = "weights/model_quantized.onnx"
    if not os.path.exists(path):
        pytest.skip(f"Quantized model file {path} not found.")
    return path


def test_onnx_model_validity(onnx_model_path):
    """Verifies that the exported ONNX model passes onnx.checker validation."""
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


def test_quantized_model_validity(quantized_model_path):
    """Verifies that the INT8 quantized ONNX model passes onnx.checker validation."""
    model = onnx.load(quantized_model_path)
    onnx.checker.check_model(model)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1


def test_numerical_alignment_pytorch_vs_onnx(onnx_model_path):
    """Verifies that PyTorch and ONNX models have numerical error < 1e-4."""
    weights_path = "weights/best_model.pth"
    if not os.path.exists(weights_path):
        pytest.skip("best_model.pth not found.")

    config = {
        "model": {
            "backbone": "efficientnet_b0",
            "num_classes": 2,
            "pretrained": False,
            "drop_rate": 0.3
        }
    }
    model = build_model(config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # Test with simulated realistic normalized images
    raw_img = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    norm_tensor = (raw_img - mean) / std

    with torch.no_grad():
        torch_out = model(norm_tensor).numpy()

    onnx_out = session.run(["output"], {"input": norm_tensor.numpy()})[0]

    max_err = np.max(np.abs(torch_out - onnx_out))
    assert max_err < 1e-4, f"Max error {max_err} exceeds 1e-4"


def test_inferencer_single_image(onnx_model_path):
    """Tests ONNXInferencer single image prediction schema and values."""
    inferencer = ONNXInferencer(
        onnx_model_path=onnx_model_path,
        threshold_config_path="configs/threshold_config.json"
    )

    # Test random uint8 image
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    res = inferencer.predict(dummy_img)

    assert isinstance(res, dict)
    assert "prediction_id" in res
    assert res["prediction_id"] in (0, 1)
    assert "label" in res
    assert res["label"] in ("Normal", "Defect")
    assert "is_defect" in res
    assert isinstance(res["is_defect"], bool)
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "defect_score" in res
    assert 0.0 <= res["defect_score"] <= 1.0
    assert "probabilities" in res
    assert "Normal" in res["probabilities"]
    assert "Defect" in res["probabilities"]
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-4


def test_inferencer_batch_images(onnx_model_path):
    """Tests ONNXInferencer batch prediction on multiple images."""
    inferencer = ONNXInferencer(onnx_model_path=onnx_model_path)
    images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(3)]

    results = inferencer.predict_batch(images)
    assert len(results) == 3
    for r in results:
        assert "prediction_id" in r
        assert "defect_score" in r
        assert isinstance(r["is_defect"], bool)


def test_inferencer_threshold_override(onnx_model_path):
    """Tests threshold override behavior in ONNXInferencer."""
    inferencer = ONNXInferencer(onnx_model_path=onnx_model_path)
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Extreme high threshold -> should not be classified as defect unless score >= 0.999
    res_high = inferencer.predict(dummy_img, threshold=0.999)
    assert res_high["threshold"] == 0.999

    # Extreme low threshold -> should be classified as defect if score >= 0.001
    res_low = inferencer.predict(dummy_img, threshold=0.001)
    assert res_low["threshold"] == 0.001


def test_inferencer_io_binding(onnx_model_path):
    """Tests ONNXInferencer with IOBinding enabled."""
    inferencer = ONNXInferencer(
        onnx_model_path=onnx_model_path,
        use_io_binding=True
    )
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = inferencer.predict(dummy_img)
    assert "label" in res
    assert "confidence" in res
    assert isinstance(res["is_defect"], bool)


def test_inferencer_quantized_model(quantized_model_path):
    """Tests ONNXInferencer with INT8 quantized model."""
    inferencer = ONNXInferencer(
        onnx_model_path=quantized_model_path,
        threshold_config_path="configs/threshold_config.json"
    )
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = inferencer.predict(dummy_img)
    assert res["prediction_id"] in (0, 1)
    assert 0.0 <= res["confidence"] <= 1.0
