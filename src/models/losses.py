from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss chống mất cân bằng dữ liệu nghiêm trọng:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - gamma: Tăng trọng số phạt cho các mẫu khó phân loại (hard examples).
    - alpha: Cân bằng tần suất giữa các class (ví dụ Normal vs Defect).
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = 0.25,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # Gán alpha nhị phân [1 - alpha, alpha]
            self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            raise TypeError(f"Kiểu dữ liệu alpha không hợp lệ: {type(alpha)}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: Logits từ mô hình có kích thước [B, C]
        targets: Ground truth nhãn [B] kiểu torch.long
        """
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # Lấy log(p_t) và p_t cho target class
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Trọng số Focal (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Alpha weight
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(inputs.device)
            at = alpha_tensor.gather(dim=0, index=targets)
            loss = -at * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Factory function tạo hàm loss dựa trên cấu hình."""
    training_cfg = config.get("training", {})
    loss_type = training_cfg.get("loss_type", "focal").lower()

    if loss_type == "focal":
        gamma = training_cfg.get("focal_gamma", 2.0)
        alpha = training_cfg.get("focal_alpha", 0.25)
        if class_weights is not None:
            alpha = class_weights
        return FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
    elif loss_type in ["weighted_ce", "cross_entropy"]:
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Loại loss không hỗ trợ: {loss_type}")
