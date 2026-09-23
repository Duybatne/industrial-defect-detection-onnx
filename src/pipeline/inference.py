import os
import sys
import json
import time
from pathlib import Path
from typing import Tuple, Union, List, Dict, Any, Optional
import numpy as np
import cv2
import onnxruntime as ort

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class ONNXInferencer:
    """
    Industrial ONNX Runtime Inference Engine:
    - Preprocessing with standard ImageNet scaling & normalization
    - AES-256 in-memory decrypted model loading support
    - Threshold calibration integration (T* from threshold_config.json)
    - Dynamic batch inference support
    - Optional IOBinding for high-throughput zero-copy memory transfers
    - Human-in-the-loop uncertainty margin monitoring (0.40 <= P <= 0.65)
    - Standardized production JSON response schema
    """
    def __init__(
        self,
        onnx_model_path: Optional[Union[str, Path]] = "weights/model.onnx",
        model_bytes: Optional[bytes] = None,
        secret_key: Optional[str] = None,
        threshold_config_path: Optional[str] = "configs/threshold_config.json",
        img_size: Tuple[int, int] = (224, 224),
        class_names: Tuple[str, ...] = ("Normal", "Defect"),
        num_threads: int = 4,
        use_io_binding: bool = False
    ):
        self.img_size = img_size
        self.class_names = class_names
        self.use_io_binding = use_io_binding

        # Configure ONNX Runtime session options
        self.opts = ort.SessionOptions()
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            self.opts.intra_op_num_threads = num_threads

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]

        # Initialize session from RAM bytes, encrypted file, or standard file
        if model_bytes is not None:
            self.session = ort.InferenceSession(model_bytes, self.opts, providers=selected_providers)
            self.model_path = "<RAM-Bytes>"
        elif onnx_model_path is not None:
            path_str = str(onnx_model_path)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"ONNX model file not found at: {path_str}")

            if path_str.endswith(".enc"):
                from src.pipeline.security import load_encrypted_model_to_ram
                key = secret_key or os.getenv("MODEL_SECRET_KEY", "industrial-secret-key-32bytes-12345")
                decrypted_bytes = load_encrypted_model_to_ram(path_str, key)
                self.session = ort.InferenceSession(decrypted_bytes, self.opts, providers=selected_providers)
                self.model_path = f"<RAM-Decrypted:{path_str}>"
            else:
                self.session = ort.InferenceSession(path_str, self.opts, providers=selected_providers)
                self.model_path = path_str
        else:
            raise ValueError("Either onnx_model_path or model_bytes must be provided.")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load decision threshold
        self.threshold = 0.5
        if threshold_config_path and os.path.exists(threshold_config_path):
            try:
                with open(threshold_config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.threshold = float(cfg.get("optimal_threshold", 0.5))
            except Exception as e:
                print(f"[!] Warning loading threshold config: {e}. Defaulting to 0.5")

    def preprocess(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Preprocesses a single image to ImageNet-standardized tensor [1, 3, H, W].
        Accepts: file path (str/Path) or numpy array (BGR or RGB).
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to read image from path: {image}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # Grayscale
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize
        resized = cv2.resize(img_rgb, self.img_size, interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        norm = resized.astype(np.float32) / 255.0
        # ImageNet mean & std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std

        # Transpose HWC -> CHW and add batch dimension -> [1, 3, H, W]
        tensor = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, ...])
        return tensor

    def preprocess_batch(self, images: List[Union[str, np.ndarray, Path]]) -> np.ndarray:
        """Preprocesses a list of images into a single batched tensor [B, 3, H, W]."""
        if not images:
            raise ValueError("Input image list is empty.")
        tensors = [self.preprocess(img) for img in images]
        return np.concatenate(tensors, axis=0)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict(
        self,
        image: Union[str, np.ndarray, Path],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes inference on a single image.

        Args:
            image: Image file path or numpy array.
            threshold: Optional custom decision threshold overriding default.

        Returns:
            Dictionary with prediction results and latency.
        """
        thresh = self.threshold if threshold is None else threshold
        t0 = time.perf_counter()
        tensor = self.preprocess(image)

        if self.use_io_binding:
            io_binding = self.session.io_binding()
            device_type = "cpu"
            io_binding.bind_cpu_input(self.input_name, tensor)
            io_binding.bind_output(self.output_name, device_type)
            self.session.run_with_iobinding(io_binding)
            logits = io_binding.copy_outputs_to_cpu()[0]
        else:
            logits = self.session.run([self.output_name], {self.input_name: tensor})[0]

        t1 = time.perf_counter()
        inference_time_ms = round((t1 - t0) * 1000.0, 2)

        probs = self._softmax(logits)[0]
        defect_score = float(probs[1])
        is_defect = bool(defect_score >= thresh)
        uncertainty_flag = bool(0.40 <= defect_score <= 0.65)
        pred_idx = 1 if is_defect else 0
        label = self.class_names[pred_idx]

        return {
            "prediction_id": pred_idx,
            "label": label,
            "is_defect": is_defect,
            "confidence": float(probs[pred_idx]),
            "defect_score": defect_score,
            "threshold": thresh,
            "inference_time_ms": inference_time_ms,
            "uncertainty_flag": uncertainty_flag,
            "probabilities": {
                self.class_names[0]: float(probs[0]),
                self.class_names[1]: float(probs[1])
            }
        }

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Path]],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes batch inference on multiple images.

        Args:
            images: List of image file paths or numpy arrays.
            threshold: Optional custom decision threshold.

        Returns:
            List of result dictionaries.
        """
        if not images:
            return []

        thresh = self.threshold if threshold is None else threshold
        t0 = time.perf_counter()
        batch_tensor = self.preprocess_batch(images)

        logits = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000.0
        per_item_time_ms = round(total_time_ms / len(images), 2)

        probs = self._softmax(logits)

        results = []
        for i in range(len(images)):
            p = probs[i]
            defect_score = float(p[1])
            is_defect = bool(defect_score >= thresh)
            uncertainty_flag = bool(0.40 <= defect_score <= 0.65)
            pred_idx = 1 if is_defect else 0
            label = self.class_names[pred_idx]

            results.append({
                "prediction_id": pred_idx,
                "label": label,
                "is_defect": is_defect,
                "confidence": float(p[pred_idx]),
                "defect_score": defect_score,
                "threshold": thresh,
                "inference_time_ms": per_item_time_ms,
                "uncertainty_flag": uncertainty_flag,
                "probabilities": {
                    self.class_names[0]: float(p[0]),
                    self.class_names[1]: float(p[1])
                }
            })

        return results
