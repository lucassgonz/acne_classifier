"""
Regenera fig4 (matriz de confusao da ablacao WithEmbed), fig5 (acuracia
por fold) e fig7 (matriz de confusao da fusao multi-regiao) para o
CBEB_final.tex, com os numeros finais da sessao de 2026-10-16/19
(Acne04+AcneLevel tratado, crops filtrados por visibilidade v3).

Uso:
    cd training
    python utils/make_paper_figures.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

OUT_DIR = Path("../papers/cbeb/figures")
GRADES = ["0", "1", "2", "3"]


def plot_confusion(cm, title, out_path):
    cm = np.array(cm, dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(6, 5.2))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=100,
                xticklabels=GRADES, yticklabels=GRADES, cbar=True,
                annot_kws={"size": 13})
    for t in plt.gca().texts:
        t.set_text(t.get_text() + "%")
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("saved", out_path)


def fig4_ablation_confusion():
    d = json.load(open("results/full_metrics_embed_ablation_v3_with.json"))
    plot_confusion(d["confusion_matrix"], "Normalized Predictions (%)", OUT_DIR / "fig4_.png")


def fig5_fold_accuracy():
    # fold accuracies for embed_ablation_v3_with, from training logs
    fold_acc = [58.35, 60.71, 57.86, 56.37, 58.52]
    plt.figure(figsize=(6, 5.2))
    plt.boxplot([fold_acc], widths=0.3, patch_artist=True,
                boxprops=dict(facecolor="#8fbf9f"), medianprops=dict(color="orange"))
    xs = np.random.default_rng(0).normal(1, 0.02, size=len(fold_acc))
    plt.scatter(xs, fold_acc, color="red", s=60, zorder=3, label="Cross-Validation Folds")
    plt.ylim(50, 70)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Across Grouped Cross-Validation Folds")
    plt.xticks([])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_.png", dpi=150)
    plt.close()
    print("saved", OUT_DIR / "fig5_.png", "mean=", np.mean(fold_acc), "std=", np.std(fold_acc))


def fig7_fusion_confusion():
    fused_labels, fused_preds = [], []
    for fold in range(5):
        p = Path("results/kfold") / "embed_ablation_v3_with" / f"fold{fold}.npz"
        d = np.load(p, allow_pickle=True)
        names, labels, probs = d["names"], d["labels"], d["probs"]
        by_name = defaultdict(list)
        for name, label, prob in zip(names, labels, probs):
            by_name[name].append((label, prob))
        for name, rows in by_name.items():
            lbls = [r[0] for r in rows]
            fused_prob = np.mean([r[1] for r in rows], axis=0)
            fused_labels.append(lbls[0])
            fused_preds.append(int(fused_prob.argmax()))

    acc = accuracy_score(fused_labels, fused_preds)
    f1m = f1_score(fused_labels, fused_preds, average="macro")
    qwk = cohen_kappa_score(fused_labels, fused_preds, weights="quadratic")
    cm = confusion_matrix(fused_labels, fused_preds, labels=[0, 1, 2, 3])
    print("fusion with_embedding: acc=", acc, "macroF1=", f1m, "qwk=", qwk)
    print(cm)

    out = {"accuracy": float(acc), "f1_macro": float(f1m), "qwk": float(qwk),
           "confusion_matrix": cm.tolist(), "n": len(fused_labels)}
    json.dump(out, open("results/full_metrics_region_fusion_with.json", "w"), indent=2)

    plot_confusion(cm, "Normalized Predictions (%)", OUT_DIR / "fig7_.png")


if __name__ == "__main__":
    fig4_ablation_confusion()
    fig5_fold_accuracy()
    fig7_fusion_confusion()
