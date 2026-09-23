import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _derive_256bit_key(key: Union[str, bytes]) -> bytes:
    """Derives a strictly 32-byte (256-bit) AES key using SHA-256."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hashlib.sha256(key).digest()


def encrypt_model(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Union[str, bytes]
) -> str:
    """
    Encrypts an ONNX model file using AES-256-GCM.
    
    Structure of encrypted payload:
    [12 bytes Nonce/IV] + [Ciphertext + 16 bytes Authentication Tag]

    Args:
        model_path: Path to plaintext .onnx model.
        output_path: Path to destination encrypted file (e.g. .onnx.enc).
        key: Secret key (string or bytes).

    Returns:
        Path to the encrypted file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Source model file not found: {model_path}")

    derived_key = _derive_256bit_key(key)
    aesgcm = AESGCM(derived_key)
    nonce = os.urandom(12)

    with open(model_path, "rb") as f:
        plaintext_bytes = f.read()

    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(nonce + ciphertext)

    print(f"[✓] Model encrypted successfully: {output_path} ({len(nonce + ciphertext)} bytes)")
    return str(output_path)


def load_encrypted_model_to_ram(
    encrypted_path: Union[str, Path],
    key: Union[str, bytes]
) -> bytes:
    """
    Reads and decrypts an encrypted model directly in RAM.
    Zero disk writing: Plaintext weights exist ONLY in volatile memory.

    Args:
        encrypted_path: Path to encrypted .onnx.enc file.
        key: Secret key used during encryption.

    Returns:
        Plaintext ONNX model bytes ready for onnxruntime.InferenceSession.
    """
    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"Encrypted model file not found: {encrypted_path}")

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    if len(encrypted_data) < 28:  # 12 bytes nonce + min 16 bytes tag
        raise ValueError("Encrypted data is corrupted or too short.")

    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]

    derived_key = _derive_256bit_key(key)
    aesgcm = AESGCM(derived_key)

    try:
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise ValueError("Decryption failed: Invalid secret key or corrupted payload.") from e

    return plaintext_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AES-256 ONNX Model Encryption Tool")
    parser.add_argument("--encrypt", help="Path to input ONNX file to encrypt")
    parser.add_argument("--output", help="Path to output encrypted file")
    parser.add_argument("--verify", help="Path to encrypted file to verify decryption in RAM")
    parser.add_argument("--key", default=os.getenv("MODEL_SECRET_KEY", "industrial-secret-key-32bytes-12345"), help="Secret encryption key")
    args = parser.parse_args()

    if args.encrypt:
        out = args.output or f"{args.encrypt}.enc"
        encrypt_model(args.encrypt, out, args.key)
    elif args.verify:
        data = load_encrypted_model_to_ram(args.verify, args.key)
        print(f"[✓] Verification SUCCESS: Decrypted {len(data)} bytes directly in RAM.")
    else:
        parser.print_help()
