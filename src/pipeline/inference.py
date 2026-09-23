from typing import Tuple, Union
import numpy as np
import cv2
import onnxruntime as ort


class ONNXInferencer:
    def __init__(self, onnx_model_path: str, img_size: Tuple[int, int] = (224, 224), class_names: Tuple[str, ...] = ("Normal", "Defect")):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size = img_size
        self.class_names = class_names

    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(image, self.img_size)
        norm = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std
        
        # [H, W, C] -> [1, C, H, W]
        tensor = np.transpose(norm, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, image: Union[str, np.ndarray]) -> dict:
        inp = self.preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: inp})[0]
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0][pred_idx])

        return {
            "prediction_id": pred_idx,
            "label": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs[0])}
        }
