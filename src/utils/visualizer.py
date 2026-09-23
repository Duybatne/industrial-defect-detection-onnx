import os
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Normal (Good)", "Defect"],
    save_path: Optional[str] = None
):
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) với số lượng mẫu và tỷ lệ chuẩn hóa."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Industrial Defect Confusion Matrix", fontweight="bold")
    plt.grid(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Vẽ Precision-Recall Curve và đánh dấu ngưỡng tối ưu T*."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="#2b5c8f", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")

    if optimal_threshold is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        plt.scatter(
            recalls[idx], precisions[idx],
            color="red", s=100, zorder=5,
            label=f"Optimal T* = {optimal_threshold:.2f} (Recall={recalls[idx]:.2f})"
        )

    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Precision-Recall Curve", fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """Vẽ Receiver Operating Characteristic (ROC) Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontweight="bold")
    plt.title("ROC Curve", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Vẽ biểu đồ quá trình học (Train/Val Loss và Train/Val F1-Score)."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    ax1.plot(epochs, history.get("train_loss", []), "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history.get("val_loss", []), "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # F1 Curve
    ax2.plot(epochs, history.get("train_f1", []), "o-", label="Train F1", color="#2ca02c")
    ax2.plot(epochs, history.get("val_f1", []), "s-", label="Val F1", color="#d62728")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("F1-Score", fontweight="bold")
    ax2.set_title("Training & Validation F1-Score", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_failure_cases(
    image_paths: List[str],
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
    class_names: List[str] = ["Good", "Defect"],
    save_path: Optional[str] = None
):
    """
    Trực quan hóa các ca dự đoán sai (False Positives và False Negatives)
    giúp phân tích rủi ro trong kiểm định chất lượng sản phẩm.
    """
    failures = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            failures.append((image_paths[i], y_true[i], y_pred[i], y_prob[i]))

    if not failures:
        print("[Visualizer] Không có ca dự đoán sai (100% Correct Predictions)!")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "100% Chính Xác - Không Có Failure Cases", 
                ha="center", va="center", fontsize=12, fontweight="bold", color="green")
        ax.axis("off")
    else:
        n = min(8, len(failures))
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for i in range(len(axes)):
            if i < n:
                img_p, true_lbl, pred_lbl, prob = failures[i]
                img = cv2.imread(img_p)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                
                case_type = "False Negative" if true_lbl == 1 else "False Positive"
                title = f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]} ({prob:.2f})\n[{case_type}]"
                axes[i].set_title(title, color="red", fontsize=9, fontweight="bold")
            axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
