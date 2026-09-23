from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import timm


class DefectClassifier(nn.Module):
    """
    Mô hình phân loại khuyết tật bề mặt sản phẩm công nghiệp:
    - Backbone Transfer Learning: EfficientNet-B0 / ResNet18 từ timm
    - Head phân loại với Dropout chống overfitting
    - Hỗ trợ đóng băng (freeze) và mở khóa (unfreeze) backbone
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        drop_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Khởi tạo backbone với num_classes=0 để lấy feature extractor
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0
        )
        
        in_features = self.model.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Đóng băng trọng số của backbone, chỉ huấn luyện classifier head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Mở khóa toàn bộ trọng số để fine-tune toàn mạng."""
        for param in self.parameters():
            param.requires_grad = True


def build_model(config: Dict[str, Any]) -> DefectClassifier:
    """Factory function khởi tạo DefectClassifier từ config dict."""
    model_cfg = config.get("model", {})
    return DefectClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        in_channels=model_cfg.get("in_channels", 3),
        drop_rate=model_cfg.get("drop_rate", 0.3)
    )
