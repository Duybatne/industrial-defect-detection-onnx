import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import create_dataloaders
from src.models import build_model, build_loss_fn
from src.utils.metrics import calculate_metrics
from src.utils.visualizer import plot_training_curves


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    max_norm: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        total_loss += loss.item() * len(targets)
        probs = torch.softmax(outputs, dim=-1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=-1).detach().cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    return epoch_loss, metrics


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện lỗi bề mặt công nghiệp")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=None, help="Ghi đè learning rate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup directories
    training_cfg = config.get("training", {})
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "weights"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_path = checkpoint_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("TrainLogger")
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 2: PYTORCH BASELINE")
    logger.info(f"Config File: {args.config}")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng thiết bị tính toán: {device}")

    # DataLoaders
    data_cfg = config.get("data", {})
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    img_size = tuple(data_cfg.get("img_size", [224, 224]))
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = min(data_cfg.get("num_workers", 2), 2) if device.type == "cpu" else data_cfg.get("num_workers", 4)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=processed_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=True
    )

    # Model
    model = build_model(config).to(device)
    logger.info(f"Đã khởi tạo kiến trúc: {config['model']['backbone']} (Pretrained={config['model']['pretrained']})")

    # Loss function (Focal Loss)
    criterion = build_loss_fn(config)
    logger.info(f"Hàm Loss: {criterion.__class__.__name__}")

    # Optimizer & Scheduler
    epochs = args.epochs or training_cfg.get("epochs", 25)
    lr = args.lr or training_cfg.get("lr", 0.0003)
    weight_decay = training_cfg.get("weight_decay", 0.0001)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = training_cfg.get("mixed_precision", True) and device.type == "cuda"

    patience = training_cfg.get("early_stopping_patience", 7)
    best_val_f1 = 0.0
    patience_counter = 0
    best_weights_path = checkpoint_dir / "best_model.pth"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": []
    }

    # Warmup head: Đóng băng backbone 2 epochs đầu
    warmup_epochs = 2
    if warmup_epochs > 0:
        logger.info(f"[Warmup] Đóng băng backbone trong {warmup_epochs} epochs đầu để khởi động classifier head...")
        model.freeze_backbone()

    for epoch in range(1, epochs + 1):
        if epoch == warmup_epochs + 1:
            logger.info("[Fine-tuning] Mở khóa toàn bộ backbone để fine-tuning end-to-end...")
            model.unfreeze_backbone()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp
        )

        val_loss, val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1_score"])
        history["val_f1"].append(val_metrics["f1_score"])

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss (Train/Val): {train_loss:.4f} / {val_loss:.4f} | "
            f"Train F1: {train_metrics['f1_score']:.3f} | "
            f"Val Recall: {val_metrics['recall']:.3f} | "
            f"Val F1: {val_metrics['f1_score']:.3f} | "
            f"LR: {curr_lr:.2e}"
        )

        # Lưu checkpoint khi đạt F1 hoặc Recall tốt hơn
        if val_metrics["f1_score"] > best_val_f1 or (val_metrics["f1_score"] == best_val_f1 and val_metrics["recall"] >= 0.98):
            best_val_f1 = val_metrics["f1_score"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["f1_score"],
                "val_recall": val_metrics["recall"],
                "val_loss": val_loss,
                "config": config
            }, str(best_weights_path))
            logger.info(f"  -> [BEST CHECKPOINT] Lưu mô hình tốt nhất với Val F1={best_val_f1:.4f} tại {best_weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[Early Stopping] Đạt ngưỡng dừng sớm (patience={patience}) tại epoch {epoch}.")
                break

    # Vẽ và lưu biểu đồ huấn luyện
    plot_training_curves(history, save_path=str(reports_dir / "training_curves.png"))
    logger.info(f"[✓] Đã xuất biểu đồ quá trình học tại reports/training_curves.png")
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH HUẤN LUYỆN")
    logger.info(f"Mô hình tốt nhất được lưu tại: {best_weights_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
