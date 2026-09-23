import os
import pytest
import numpy as np

from src.pipeline.security import encrypt_model, load_encrypted_model_to_ram
from src.pipeline.inference import ONNXInferencer


def test_encrypt_and_decrypt_round_trip(tmp_path):
    """Verifies AES-256 round-trip encryption and RAM decryption."""
    # Create dummy binary model payload
    original_data = b"DUMMY_ONNX_MODEL_BINARY_PAYLOAD_FOR_TESTING_1234567890" * 50
    plain_file = tmp_path / "model.onnx"
    plain_file.write_bytes(original_data)

    enc_file = tmp_path / "model.onnx.enc"
    key = "test-super-secret-key-for-unit-test"

    # Encrypt
    encrypted_path = encrypt_model(plain_file, enc_file, key)
    assert os.path.exists(encrypted_path)
    assert enc_file.read_bytes() != original_data

    # Decrypt in RAM
    decrypted_bytes = load_encrypted_model_to_ram(encrypted_path, key)
    assert decrypted_bytes == original_data


def test_decrypt_with_wrong_key(tmp_path):
    """Verifies that decryption fails when provided an incorrect secret key."""
    plain_file = tmp_path / "model.onnx"
    plain_file.write_bytes(b"SECRET_MODEL_CONTENT")

    enc_file = tmp_path / "model.onnx.enc"
    encrypt_model(plain_file, enc_file, key="correct-key-12345")

    with pytest.raises(ValueError, match="Decryption failed"):
        _ = load_encrypted_model_to_ram(enc_file, key="wrong-key-99999")


def test_decrypt_corrupted_payload(tmp_path):
    """Verifies that corrupted or truncated encrypted files are rejected."""
    corrupted_file = tmp_path / "corrupted.onnx.enc"
    corrupted_file.write_bytes(b"TOO_SHORT")

    with pytest.raises(ValueError, match="corrupted or too short"):
        _ = load_encrypted_model_to_ram(corrupted_file, key="any-key")


def test_inferencer_from_ram_bytes():
    """Verifies that ONNXInferencer can initialize directly from RAM bytes."""
    onnx_path = "weights/model.onnx"
    if not os.path.exists(onnx_path):
        pytest.skip(f"{onnx_path} not found.")

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()

    inferencer = ONNXInferencer(model_bytes=model_bytes)
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = inferencer.predict(dummy_img)

    assert res["prediction_id"] in (0, 1)
    assert "confidence" in res
    assert "uncertainty_flag" in res
