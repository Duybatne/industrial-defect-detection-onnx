import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DefectDataset, get_transforms
from src.models import build_model
from src.utils.metrics import calculate_metrics, find_optimal_threshold
from src.utils.visualizer import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_failure_cases
)


def run_inference_on_dataset(
    model: nn.Module,
    dataset: DefectDataset,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    return np.array(all_targets), np.array(all_probs), dataset.image_paths


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình và tinh chỉnh ngưỡng quyết định (Threshold Calibration)")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Đường dẫn file checkpoint weights")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình train")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Tập dữ liệu cần đánh giá")
    parser.add_argument("--target_recall", type=float, default=0.98, help="Ngưỡng Recall mục tiêu cho lớp Defect")
    parser.add_argument("--output_dir", type=str, default="reports", help="Thư mục lưu báo cáo kết quả")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Sử dụng thiết bị: {device}")

    # Nạp mô hình và checkpoint
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {weights_path}")

    model = build_model(config).to(device)
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"[✓] Đã nạp thành công checkpoint từ: {weights_path}")

    # Nạp dữ liệu tập cần đánh giá
    data_cfg = config.get("data", {})
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    img_size = tuple(data_cfg.get("img_size", [224, 224]))
    split_dir = os.path.join(processed_dir, args.split)

    transform = get_transforms(img_size=img_size, is_train=False)
    dataset = DefectDataset.from_directory(split_dir, transform=transform)
    print(f"[+] Đã nạp tập '{args.split}': Tổng {len(dataset)} ảnh")

    # Dự đoán
    y_true, y_prob, image_paths = run_inference_on_dataset(model, dataset, device)

    # 1. Tính toán metrics mặc định (T = 0.5)
    y_pred_default = (y_prob >= 0.5).astype(int)
    default_metrics = calculate_metrics(y_true, y_pred_default, y_prob)

    # 2. Tinh chỉnh ngưỡng (Threshold Calibration)
    optimal_t, opt_metrics = find_optimal_threshold(y_true, y_prob, target_recall=args.target_recall)
    y_pred_opt = (y_prob >= optimal_t).astype(int)

    # In kết quả so sánh
    print("\n" + "=" * 65)
    print(f"BẢNG TỔNG HỢP HIỆU NĂNG TRÊN TẬP {args.split.upper()}")
    print("=" * 65)
    print(f"{'CHỈ SỐ METRIC':<20} | {'MẶC ĐỊNH (T=0.5)':<18} | {'TỐI ƯU (T*=' + f'{optimal_t:.2f})':<20}")
    print("-" * 65)
    print(f"{'Defect Recall':<20} | {default_metrics['recall']*100:<17.2f}% | {opt_metrics['recall']*100:<19.2f}%")
    print(f"{'Precision':<20} | {default_metrics['precision']*100:<17.2f}% | {opt_metrics['precision']*100:<19.2f}%")
    print(f"{'F1-Score':<20} | {default_metrics['f1_score']:<18.4f} | {opt_metrics['f1_score']:<20.4f}")
    print(f"{'Accuracy':<20} | {default_metrics['accuracy']*100:<17.2f}% | {opt_metrics['accuracy']*100:<19.2f}%")
    print(f"{'Specificity':<20} | {default_metrics['specificity']*100:<17.2f}% | {opt_metrics['specificity']*100:<19.2f}%")
    print(f"{'ROC-AUC':<20} | {default_metrics.get('roc_auc', 0.0):<18.4f} | {opt_metrics.get('roc_auc', 0.0):<20.4f}")
    print(f"{'PR-AUC':<20} | {default_metrics.get('pr_auc', 0.0):<18.4f} | {opt_metrics.get('pr_auc', 0.0):<20.4f}")
    print("=" * 65 + "\n")

    # Kiểm tra tiêu chuẩn chất lượng (Quality Gates)
    pass_recall = opt_metrics["recall"] >= args.target_recall
    pass_f1 = opt_metrics["f1_score"] >= 0.92

    print("[QA Check] TIÊU CHUẨN ĐÁNH GIÁ GIAI ĐOẠN 2:")
    print(f"  - Defect Recall >= {args.target_recall*100:.0f}%: {'[PASS]' if pass_recall else '[FAIL]'} ({opt_metrics['recall']*100:.1f}%)")
    print(f"  - F1-Score >= 0.92: {'[PASS]' if pass_f1 else '[FAIL]'} ({opt_metrics['f1_score']:.3f})")

    # Lưu cấu hình ngưỡng tối ưu
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    threshold_config = {
        "optimal_threshold": optimal_t,
        "target_recall": args.target_recall,
        "split_evaluated": args.split,
        "metrics_at_optimal_threshold": opt_metrics
    }
    threshold_cfg_path = Path("configs/threshold_config.json")
    with open(threshold_cfg_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"[✓] Đã lưu cấu hình ngưỡng tối ưu tại: {threshold_cfg_path}")

    # Lưu summary metrics JSON
    metrics_summary_path = out_dir / "metrics_summary.json"
    with open(metrics_summary_path, "w") as f:
        json.dump({
            "split": args.split,
            "default_threshold": 0.5,
            "optimal_threshold": optimal_t,
            "default_metrics": default_metrics,
            "optimal_metrics": opt_metrics
        }, f, indent=2)
    print(f"[✓] Đã lưu tóm tắt metrics tại: {metrics_summary_path}")

    # Vẽ và lưu các biểu đồ báo cáo
    cm_path = str(out_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_opt, class_names=["Good", "Defect"], save_path=cm_path)
    print(f"[✓] Đã lưu Confusion Matrix tại: {cm_path}")

    pr_path = str(out_dir / "pr_curve.png")
    plot_pr_curve(y_true, y_prob, optimal_threshold=optimal_t, save_path=pr_path)
    print(f"[✓] Đã lưu PR Curve tại: {pr_path}")

    roc_path = str(out_dir / "roc_curve.png")
    plot_roc_curve(y_true, y_prob, save_path=roc_path)
    print(f"[✓] Đã lưu ROC Curve tại: {roc_path}")

    fail_path = str(out_dir / "failure_cases.png")
    plot_failure_cases(
        image_paths=image_paths,
        y_true=y_true.tolist(),
        y_pred=y_pred_opt.tolist(),
        y_prob=y_prob.tolist(),
        save_path=fail_path
    )
    print(f"[✓] Đã lưu Failure Cases analysis tại: {fail_path}")


if __name__ == "__main__":
    main()
