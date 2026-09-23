import argparse
import os
import yaml
import torch
from src.models.network import build_model


def export_to_onnx(config_path: str, output_path: str = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if output_path is None:
        output_path = config.get("export", {}).get("onnx_path", "weights/model.onnx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = build_model(config)
    model.eval()

    img_size = config.get("data", {}).get("img_size", [224, 224])
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], requires_grad=False)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if config.get("export", {}).get("dynamic_axes", True) else None
    opset = config.get("export", {}).get("opset_version", 17)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    print(f"Exported ONNX model successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    args = parser.parse_args()
    export_to_onnx(args.config, args.output)
