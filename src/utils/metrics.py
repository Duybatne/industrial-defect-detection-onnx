from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Tính toán toàn diện các chỉ số phân loại cho bài toán Defect Detection:
    Accuracy, Precision, Recall (Defect), F1-Score, Specificity, ROC-AUC, PR-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    # Tính specificity: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.98,
    threshold_step: float = 0.01
) -> Tuple[float, Dict[str, float]]:
    """
    Quét và tinh chỉnh ngưỡng quyết định (Threshold Calibration):
    Tìm ngưỡng T* sao cho Defect Recall >= target_recall (98%) với F1-Score / Precision cao nhất.
    """
    thresholds = np.arange(0.01, 1.0, threshold_step)
    best_threshold = 0.5
    best_metrics = calculate_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
    
    candidates = []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        m = calculate_metrics(y_true, pred, y_prob)
        if m["recall"] >= target_recall:
            candidates.append((t, m))

    if candidates:
        # Chọn candidate có F1 cao nhất; nếu bằng nhau chọn Precision cao nhất
        candidates.sort(key=lambda x: (x[1]["f1_score"], x[1]["precision"]), reverse=True)
        best_threshold, best_metrics = candidates[0]
    else:
        # Nếu không đạt target_recall tuyệt đối, chọn ngưỡng cho Recall cao nhất
        all_results = [(t, calculate_metrics(y_true, (y_prob >= t).astype(int), y_prob)) for t in thresholds]
        all_results.sort(key=lambda x: (x[1]["recall"], x[1]["f1_score"]), reverse=True)
        best_threshold, best_metrics = all_results[0]

    return float(best_threshold), best_metrics
