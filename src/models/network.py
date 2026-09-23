import torch
import torch.nn as nn
import timm


class DefectClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    return DefectClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True)
    )
