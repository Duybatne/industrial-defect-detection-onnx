from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_predictions(images: np.ndarray, y_true: List[str], y_pred: List[str], save_path: Optional[str] = None):
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(images[i])
            color = "green" if y_true[i] == y_pred[i] else "red"
            axes[i].set_title(f"T: {y_true[i]}\nP: {y_pred[i]}", color=color)
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
