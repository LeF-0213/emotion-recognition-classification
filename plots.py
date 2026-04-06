import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C


def plot_learning_curves(histories, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for hist, name, c in zip(histories, names, colors):
        e = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(e, hist["train_loss"], color=c, linestyle="-", label=f"{name} train")
        axes[0].plot(e, hist["val_loss"], color=c, linestyle="--", label=f"{name} val")
        axes[1].plot(e, hist["train_acc"], color=c, linestyle="-", label=f"{name} train")
        axes[1].plot(e, hist["val_acc"], color=c, linestyle="--", label=f"{name} val")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results_dict, path):
    from sklearn.metrics import confusion_matrix

    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results_dict.items()):
        cm = confusion_matrix(res["labels"], res["preds"], labels=list(range(C.NUM_CLASSES)))
        cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            cm_n,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=C.EMOTION_CLASSES,
            yticklabels=C.EMOTION_CLASSES,
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results_table(rows, path):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)