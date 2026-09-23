import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import onnx

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.network import build_model


def export_to_onnx(
    config_path: str = "configs/train_config.yaml",
    weights_path: str = "weights/best_model.pth",
    output_path: str = "weights/model.onnx",
    opset_version: int = 17,
    dynamic_batch: bool = True
) -> str:
    """
    Exports a trained PyTorch DefectClassifier model to ONNX format.

    Args:
        config_path: Path to YAML training/model config.
        weights_path: Path to PyTorch model weights (.pth).
        output_path: Target path for the exported .onnx file.
        opset_version: ONNX Opset version (default: 17).
        dynamic_batch: Whether to allow variable batch size on dim 0.

    Returns:
        The absolute path to the exported ONNX file.
    """
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "backbone": "efficientnet_b0",
                "num_classes": 2,
                "pretrained": False,
                "dropout": 0.3
            },
            "data": {"img_size": [224, 224]}
        }

    # Prepare model
    model = build_model(config)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[✓] Loaded checkpoint weights from: {weights_path}")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print(f"[✓] Loaded state_dict from: {weights_path}")
        else:
            raise ValueError(f"Invalid checkpoint format in: {weights_path}")
    else:
        print(f"[!] Warning: weights file {weights_path} not found. Exporting model with initialized weights.")

    model.eval()

    # Image size & dummy input
    img_size = config.get("data", {}).get("img_size", [224, 224])
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], requires_grad=False)

    # Dynamic axes configuration
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Exporting model to ONNX (Opset {opset_version}, Dynamic={dynamic_batch})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    # Verify exported model with onnx.checker
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"[✓] Successfully exported and verified ONNX model at: {output_path}")
    print(f"    - File size: {file_size_mb:.2f} MB")
    print(f"    - Input shape: {[None if dynamic_batch else 1, 3, img_size[0], img_size[1]]}")
    print(f"    - Output shape: {[None if dynamic_batch else 1, config.get('model', {}).get('num_classes', 2)]}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch DefectClassifier to ONNX")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--weights", default="weights/best_model.pth", help="Path to PyTorch weights")
    parser.add_argument("--output", default="weights/model.onnx", help="Path to output ONNX file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--static", action="store_true", help="Fix batch size to 1 (disable dynamic batch)")
    args = parser.parse_args()

    export_to_onnx(
        config_path=args.config,
        weights_path=args.weights,
        output_path=args.output,
        opset_version=args.opset,
        dynamic_batch=not args.static
    )
