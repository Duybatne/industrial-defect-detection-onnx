import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_transforms
from src.utils.metrics import calculate_metrics


class DefectCalibrationDataReader(CalibrationDataReader):
    """
    Calibration Data Reader for ONNX Runtime Static Quantization.
    Loads real validation images to compute accurate activation dynamic ranges.
    """
    def __init__(
        self,
        data_dir: str = "data/processed/val",
        input_name: str = "input",
        img_size: tuple = (224, 224),
        max_samples: int = 50
    ):
        self.input_name = input_name
        self.img_size = img_size
        self.transforms = get_transforms(img_size=img_size, is_train=False)

        # Collect images
        image_paths = []
        if os.path.exists(data_dir):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                image_paths.extend(Path(data_dir).rglob(ext))

        self.image_paths = sorted(image_paths)[:max_samples]
        self.enum_data = iter(self._load_samples())

    def _load_samples(self):
        for img_path in self.image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            transformed = self.transforms(image=img_rgb)["image"]
            input_tensor = transformed.unsqueeze(0).numpy()  # [1, 3, 224, 224]
            yield {self.input_name: input_tensor}

    def get_next(self) -> Optional[dict]:
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self._load_samples())


def evaluate_onnx_accuracy(
    onnx_path: str,
    val_dir: str = "data/processed/val",
    threshold: float = 0.05
) -> Dict[str, float]:
    """Evaluates an ONNX model on validation images."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    transforms = get_transforms(img_size=(224, 224), is_train=False)

    y_true = []
    y_prob = []

    # Read good (0) and defect (1)
    for label_name, label_id in [("good", 0), ("defect", 1)]:
        folder = Path(val_dir) / label_name
        if not folder.exists():
            continue
        for img_path in folder.glob("*.png"):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = transforms(image=rgb)["image"].unsqueeze(0).numpy()
            logits = session.run([output_name], {input_name: inp})[0]
            # Softmax
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            defect_prob = float(probs[0, 1])

            y_true.append(label_id)
            y_prob.append(defect_prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return calculate_metrics(y_true, y_pred, y_prob)


def quantize_model(
    input_model_path: str = "weights/model.onnx",
    output_model_path: str = "weights/model_quantized.onnx",
    val_data_dir: str = "data/processed/val",
    mode: str = "dynamic",
    threshold: float = 0.05
) -> str:
    """
    Quantizes an ONNX model to INT8 precision.

    Args:
        input_model_path: Path to FP32 ONNX model.
        output_model_path: Path to save INT8 quantized model.
        val_data_dir: Path to validation set for calibration / evaluation.
        mode: 'dynamic' or 'static'.
        threshold: Optimal decision threshold.

    Returns:
        Path to the quantized model.
    """
    print("=" * 68)
    print(f"LƯỢNG TỬ HÓA MÔ HÌNH ONNX: INT8 ({mode.upper()})")
    print("=" * 68)

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    # 1. Pre-evaluation of FP32 baseline
    print("[*] Đang đánh giá baseline ONNX FP32 trên tập Validation...")
    fp32_metrics = evaluate_onnx_accuracy(input_model_path, val_dir=val_data_dir, threshold=threshold)
    fp32_size_mb = os.path.getsize(input_model_path) / (1024 * 1024)
    print(f"    - FP32 Model Size: {fp32_size_mb:.2f} MB")
    print(f"    - FP32 Recall: {fp32_metrics['recall'] * 100:.2f}% | F1: {fp32_metrics['f1_score']:.4f}")

    # 2. Perform Quantization
    print(f"[*] Tiến hành lượng tử hóa ({mode})...")
    if mode == "static":
        session = ort.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        calib_reader = DefectCalibrationDataReader(
            data_dir=val_data_dir,
            input_name=input_name,
            img_size=(224, 224),
            max_samples=50
        )
        try:
            quantize_static(
                model_input=input_model_path,
                model_output=output_model_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8
            )
            print("[✓] Static Quantization hoàn tất.")
        except Exception as e:
            print(f"[!] Lỗi Static Quantization: {e}. Fallback sang Dynamic Quantization.")
            quantize_dynamic(
                model_input=input_model_path,
                model_output=output_model_path,
                op_types_to_quantize=["Gemm", "MatMul"],
                weight_type=QuantType.QInt8
            )
            print("[✓] Dynamic Quantization (Fallback) hoàn tất.")
    else:
        # Dynamic quantization on linear/Gemm layers
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            op_types_to_quantize=["Gemm", "MatMul"],
            weight_type=QuantType.QInt8
        )
        print("[✓] Dynamic Quantization hoàn tất.")

    # 3. Check quantized model with onnx.checker
    quant_model = onnx.load(output_model_path)
    onnx.checker.check_model(quant_model)
    quant_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)

    # 4. Post-evaluation of Quantized Model
    print("[*] Đang đánh giá mô hình INT8 trên tập Validation...")
    int8_metrics = evaluate_onnx_accuracy(output_model_path, val_dir=val_data_dir, threshold=threshold)

    recall_drop = (fp32_metrics["recall"] - int8_metrics["recall"]) * 100
    f1_drop = (fp32_metrics["f1_score"] - int8_metrics["f1_score"]) * 100

    print("=" * 68)
    print("BẢNG SO SÁNH TRƯỚC VÀ SAU LƯỢNG TỬ HÓA INT8")
    print("=" * 68)
    print(f"{'CHỈ SỐ METRIC':<25} | {'ONNX FP32':<15} | {'ONNX INT8':<15} | {'THAY ĐỔI'}")
    print("-" * 68)
    print(f"{'Model Size (MB)':<25} | {fp32_size_mb:<15.2f} | {quant_size_mb:<15.2f} | {(fp32_size_mb - quant_size_mb):.2f} MB")
    print(f"{'Defect Recall':<25} | {fp32_metrics['recall']*100:<14.2f}% | {int8_metrics['recall']*100:<14.2f}% | {-recall_drop:+.2f}%")
    print(f"{'Defect Precision':<25} | {fp32_metrics['precision']*100:<14.2f}% | {int8_metrics['precision']*100:<14.2f}% | {-(fp32_metrics['precision']-int8_metrics['precision'])*100:+.2f}%")
    print(f"{'F1-Score':<25} | {fp32_metrics['f1_score']:<15.4f} | {int8_metrics['f1_score']:<15.4f} | {-f1_drop:+.4f}")
    print(f"{'Accuracy':<25} | {fp32_metrics['accuracy']*100:<14.2f}% | {int8_metrics['accuracy']*100:<14.2f}% | {-(fp32_metrics['accuracy']-int8_metrics['accuracy'])*100:+.2f}%")
    print("=" * 68)

    # Acceptance criterion: Recall drop <= 1.0% and F1 drop <= 1.0%
    if recall_drop > 1.0 or f1_drop > 1.0:
        print(f"[!] CẢNH BÁO: Độ suy giảm F1 ({f1_drop:.2f}%) hoặc Recall ({recall_drop:.2f}%) vượt ngưỡng 1%!")
    else:
        print(f"[✓] NGHIỆM THU: Độ suy giảm chất lượng <= 1% (Recall drop: {recall_drop:.2f}%, F1 drop: {f1_drop:.2f}%). ĐẠT CHUẨN!")

    return output_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", default="weights/model.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", default="weights/model_quantized.onnx", help="Path to output INT8 model")
    parser.add_argument("--val-dir", default="data/processed/val", help="Path to validation directory")
    parser.add_argument("--mode", default="dynamic", choices=["dynamic", "static"], help="Quantization mode")
    parser.add_argument("--threshold", type=float, default=0.05, help="Decision threshold")
    args = parser.parse_args()

    quantize_model(
        input_model_path=args.input,
        output_model_path=args.output,
        val_data_dir=args.val_dir,
        mode=args.mode,
        threshold=args.threshold
    )
