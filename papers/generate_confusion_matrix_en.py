import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CM = np.array([
    [301, 15, 1, 1],
    [13, 386, 13, 0],
    [0, 9, 101, 0],
    [0, 0, 0, 250],
])

LABELS = ["0", "1", "2", "3"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_confusion_matrix(cm, normalize=False, title="", cmap="Blues", fmt=".2f", cbar_label="Count"):
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, where=row_sums != 0) * 100
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig1 = plot_confusion_matrix(
        CM,
        normalize=False,
        title="Absolute Prediction Counts",
        cmap="Blues",
        fmt=".0f",
        cbar_label="Count",
    )
    fig1.savefig(os.path.join(OUT_DIR, "confusion_matrix_counts_en.png"), dpi=300, bbox_inches="tight")
    fig1.savefig(os.path.join(OUT_DIR, "confusion_matrix_counts_en.pdf"), bbox_inches="tight")

    fig2 = plot_confusion_matrix(
        CM,
        normalize=True,
        title="Normalized Predictions (%)",
        cmap="Greens",
        cbar_label="Percentage",
    )
    fig2.savefig(os.path.join(OUT_DIR, "confusion_matrix_normalized_en.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(OUT_DIR, "confusion_matrix_normalized_en.pdf"), bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    sns.heatmap(
        CM,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Absolute Prediction Counts")

    norm = np.divide(CM.astype(float), CM.sum(axis=1, keepdims=True), where=CM.sum(axis=1, keepdims=True) != 0) * 100
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=axes[1],
        cbar_kws={"label": "Percentage"},
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title("Normalized Predictions (%)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figure4_confusion_matrix_en.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "figure4_confusion_matrix_en.pdf"), bbox_inches="tight")
    plt.close("all")
    print("Saved figures to", OUT_DIR)
