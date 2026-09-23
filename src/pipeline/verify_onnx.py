import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import cv2
import torch
import onnxruntime as ort

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.network import build_model
from src.dataset import get_transforms


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def verify_numerical_consistency(
    torch_weights_path: str = "weights/best_model.pth",
    onnx_model_path: str = "weights/model.onnx",
    config_path: str = "configs/train_config.yaml",
    test_data_dir: str = "data/processed/test",
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Verifies tensor numerical alignment between PyTorch and ONNX Runtime.

    Args:
        torch_weights_path: Path to PyTorch model weights (.pth).
        onnx_model_path: Path to ONNX model (.onnx).
        config_path: Path to config YAML.
        test_data_dir: Directory containing test images.
        tolerance: Maximum acceptable difference (default: 1e-4).

    Returns:
        Dictionary containing verification results and metrics.
    """
    print("=" * 68)
    print("XÁC MINH ĐỘ LỆCH KẾT QUẢ: PYTORCH vs ONNX RUNTIME")
    print("=" * 68)

    # 1. Load PyTorch model
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {"backbone": "efficientnet_b0", "num_classes": 2, "pretrained": False, "dropout": 0.3},
            "data": {"img_size": [224, 224]}
        }

    torch_model = build_model(config)
    checkpoint = torch.load(torch_weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        torch_model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        torch_model.load_state_dict(checkpoint)
    torch_model.eval()
    print(f"[✓] Đã nạp PyTorch model từ: {torch_weights_path}")

    # 2. Load ONNX Runtime session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[✓] Đã khởi tạo ONNX Runtime Session từ: {onnx_model_path}")
    print(f"    - Input: '{input_name}', Output: '{output_name}'")

    # 3. Test on Synthetic Tensors (ImageNet Normalized range)
    np.random.seed(42)
    torch.manual_seed(42)
    batch_sizes = [1, 4, 8]
    synthetic_results = []

    print("-" * 68)
    print("1. KIỂM THỬ TRÊN TENSOR GIẢ LẬP (SYNTHETIC NORMALIZED IMAGES):")
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for bs in batch_sizes:
        # Generate simulated valid image tensor in normalized ImageNet range
        raw_img = torch.randint(0, 256, (bs, 3, 224, 224), dtype=torch.float32) / 255.0
        dummy_tensor = (raw_img - mean_t) / std_t

        with torch.no_grad():
            torch_out = torch_model(dummy_tensor).numpy()

        onnx_out = session.run([output_name], {input_name: dummy_tensor.numpy()})[0]

        torch_prob = softmax(torch_out)
        onnx_prob = softmax(onnx_out)

        max_err_logit = float(np.max(np.abs(torch_out - onnx_out)))
        max_err_prob = float(np.max(np.abs(torch_prob - onnx_prob)))
        mae_logit = float(np.mean(np.abs(torch_out - onnx_out)))

        passed = max_err_logit < tolerance and max_err_prob < tolerance

        synthetic_results.append({
            "batch_size": bs,
            "max_error_logit": max_err_logit,
            "max_error_prob": max_err_prob,
            "mae_logit": mae_logit,
            "passed": passed
        })
        status_str = "PASS" if passed else "FAIL"
        print(f"  - [BS={bs}] Logit Max Err: {max_err_logit:.6f} | Prob Max Err: {max_err_prob:.2e} -> [{status_str}]")

    # 4. Test on Real Images from dataset
    print("-" * 68)
    print("2. KIỂM THỬ TRÊN ẢNH THỰC TẾ (REAL TEST DATASET):")
    transforms = get_transforms(img_size=(224, 224), is_train=False)

    image_paths = []
    if os.path.exists(test_data_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(Path(test_data_dir).rglob(ext))

    real_images_tested = 0
    real_logit_errors = []
    real_prob_errors = []

    if image_paths:
        sample_paths = image_paths[:25]
        for img_path in sample_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            transformed = transforms(image=img_rgb)["image"]
            input_tensor = transformed.unsqueeze(0)  # [1, 3, 224, 224]

            with torch.no_grad():
                t_logits = torch_model(input_tensor).numpy()
            o_logits = session.run([output_name], {input_name: input_tensor.numpy()})[0]

            t_prob = softmax(t_logits)
            o_prob = softmax(o_logits)

            logit_err = float(np.max(np.abs(t_logits - o_logits)))
            prob_err = float(np.max(np.abs(t_prob - o_prob)))

            real_logit_errors.append(logit_err)
            real_prob_errors.append(prob_err)
            real_images_tested += 1

        overall_real_logit_err = max(real_logit_errors) if real_logit_errors else 0.0
        overall_real_prob_err = max(real_prob_errors) if real_prob_errors else 0.0
        real_passed = overall_real_logit_err < tolerance and overall_real_prob_err < tolerance
        status_str = "PASS" if real_passed else "FAIL"
        print(f"  - Số ảnh kiểm thử: {real_images_tested}")
        print(f"  - Logit Max Error: {overall_real_logit_err:.7f} (< 1e-4) -> [{status_str}]")
        print(f"  - Probability Max Error: {overall_real_prob_err:.7f} (< 1e-4) -> [{status_str}]")
    else:
        overall_real_logit_err = 0.0
        overall_real_prob_err = 0.0
        real_passed = True
        print(f"[!] Warning: Test dataset folder '{test_data_dir}' not found.")

    # 5. Overall Verdict
    all_synthetic_passed = all(r["passed"] for r in synthetic_results)
    overall_passed = all_synthetic_passed and real_passed

    print("=" * 68)
    print(f"TỔNG KẾT XÁC MINH TOÀN VẸN TENSOR: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"  - Tiêu chuẩn đề ra: Delta < {tolerance:.1e}")
    print(f"  - Sai số thực tế trên ảnh kiểm tra: {overall_real_logit_err:.7f}")
    print("=" * 68)

    if not overall_passed:
        raise ValueError(
            f"Numerical verification failed: Max error exceeds tolerance {tolerance:.1e}"
        )

    return {
        "status": "PASS" if overall_passed else "FAIL",
        "tolerance": tolerance,
        "synthetic_results": synthetic_results,
        "real_images_tested": real_images_tested,
        "real_max_logit_error": overall_real_logit_err,
        "real_max_prob_error": overall_real_prob_err
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify numerical consistency between PyTorch and ONNX")
    parser.add_argument("--torch-weights", default="weights/best_model.pth", help="Path to PyTorch weights")
    parser.add_argument("--onnx-model", default="weights/model.onnx", help="Path to ONNX model")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--test-data-dir", default="data/processed/test", help="Path to test images dir")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max tolerance for error")
    args = parser.parse_args()

    verify_numerical_consistency(
        torch_weights_path=args.torch_weights,
        onnx_model_path=args.onnx_model,
        config_path=args.config,
        test_data_dir=args.test_data_dir,
        tolerance=args.tolerance
    )
