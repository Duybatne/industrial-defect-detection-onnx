import io
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from deployment.app import app, API_KEY


@pytest.fixture(scope="session", autouse=True)
def ensure_onnx_model():
    """Ensures at least one ONNX model exists before running API tests in any environment."""
    os.makedirs("weights", exist_ok=True)
    target = "weights/model.onnx"
    if not (os.path.exists(target) or os.path.exists("weights/model_quantized.onnx") or os.path.exists("weights/model.onnx.enc")):
        from deployment.export_onnx import export_to_onnx
        export_to_onnx(output_path=target)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_png_bytes():
    """Generates a valid encoded PNG image in memory."""
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())


def test_health_endpoint(client):
    """Verifies that GET /health responds with 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_predict_single_image_unauthorized(client, sample_png_bytes):
    """Verifies that missing API key returns 401 Unauthorized."""
    response = client.post(
        "/v1/predict",
        files={"file": ("test.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 401
    assert "Invalid or missing X-API-Key" in response.json()["detail"]


def test_predict_single_image_authorized(client, sample_png_bytes):
    """Verifies authorized prediction on a valid PNG image."""
    response = client.post(
        "/v1/predict",
        headers={"X-API-Key": API_KEY},
        files={"file": ("test.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()

    assert "prediction_id" in data
    assert data["prediction_id"] in (0, 1)
    assert "label" in data
    assert data["label"] in ("Normal", "Defect")
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "Normal" in data["probabilities"]
    assert "Defect" in data["probabilities"]
    assert "inference_time_ms" in data
    assert isinstance(data["uncertainty_flag"], bool)


def test_predict_batch_images(client):
    """Verifies batch image classification via /v1/predict-batch."""
    img1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    _, buf1 = cv2.imencode(".png", img1)
    _, buf2 = cv2.imencode(".png", img2)

    files = [
        ("files", ("img1.png", io.BytesIO(buf1.tobytes()), "image/png")),
        ("files", ("img2.png", io.BytesIO(buf2.tobytes()), "image/png"))
    ]

    response = client.post(
        "/v1/predict-batch",
        headers={"X-API-Key": API_KEY},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
    assert "total_inference_time_ms" in data


def test_predict_invalid_file_type(client):
    """Verifies that non-image file uploads return 400 Bad Request."""
    text_content = io.BytesIO(b"Hello world this is not an image")
    response = client.post(
        "/v1/predict",
        headers={"X-API-Key": API_KEY},
        files={"file": ("test.txt", text_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_corrupted_image(client):
    """Verifies that corrupted images return 400 Bad Request."""
    corrupted_data = io.BytesIO(b"\x89PNG\r\n\x1a\nCorruptedGarbageBytes123456789")
    response = client.post(
        "/v1/predict",
        headers={"X-API-Key": API_KEY},
        files={"file": ("corrupt.png", corrupted_data, "image/png")}
    )
    assert response.status_code == 400
    assert "Could not decode image" in response.json()["detail"]
