"""
Numeros finais do modelo de imagem inteira: media 50/50 (peso fixado a priori, sem ajuste) entre
  A = receita do artigo (ResNet-18 + CBAM + SCIN + mixup + SWA + TTA, ensemble de seeds) e
  B = receita com supervisao de contagem de lesoes (ResNet-18, run_jagc_original_replica.py, ensemble de seeds),
nos dois protocolos: dataset combinado (n=2863) e split oficial do ACNE04 (n=1460). Tambem reporta
cada grupo sozinho. Salva results/final_whole_ensemble.json e a matriz de confusao normalizada do combinado.

Uso: python evaluate_final_whole_ensemble.py [--fig ../papers/cbeb/figures/fig8_ensemble.png]
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

R = "results/kfold_wholeimage"
PROTOCOLS = {
    "combined": {
        "A": ["whole_v2_cbam_swa_tta"] + [f"whole_v2_cbam_swa_tta_seed{s}" for s in (43, 44, 45, 46)],
        "B": ["jagc_replica_combined_resnet18"] + [f"jagc_replica_combined_resnet18_seed{s}" for s in range(43, 49)],
    },
    "official": {
        "A": ["whole_official_cbam_swa_tta"],
        "B": ["jagc_replica_resnet18"] + [f"jagc_replica_resnet18_seed{s}" for s in (43, 44, 45, 46)],
    },
}


def load(d, k):
    z = np.load(os.path.join(R, d, f"fold{k}.npz"), allow_pickle=True)
    return dict(zip(z["names"], z["probs"])), dict(zip(z["names"], z["labels"]))


def ece(probs, y, bins=15):
    conf, hit = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += abs(hit[m].mean() - conf[m].mean()) * m.mean()
    return out


def metrics(probs, y):
    p = probs.argmax(1)
    cm = confusion_matrix(y, p, labels=range(4))
    n = cm.sum()
    spec = [(n - cm[c].sum() - cm[:, c].sum() + cm[c, c]) / (n - cm[c].sum()) for c in range(4)]
    return {
        "n": int(len(y)), "acc": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro"),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"), "ece": ece(probs, y),
        "sensitivity": (cm.diagonal() / cm.sum(1)).tolist(), "specificity": spec,
        "support": cm.sum(1).tolist(), "confusion": cm.tolist(),
    }


def evaluate(groups):
    ys, pa, pb = [], [], []
    for k in range(5):
        first, labels = load(groups["B"][0], k)
        names = sorted(first)
        ys.append(np.array([labels[n] for n in names]))
        for G, acc in (("A", pa), ("B", pb)):
            acc.append(np.mean([[load(d, k)[0][n] for n in names] for d in groups[G]], axis=0))
    y, pa, pb = np.concatenate(ys), np.concatenate(pa), np.concatenate(pb)
    return y, {"A": metrics(pa, y), "B": metrics(pb, y), "mix_50_50": metrics(0.5 * pa + 0.5 * pb, y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None, help="caminho do PNG da matriz de confusao (combinado, mistura)")
    ap.add_argument("--out", default="results/final_whole_ensemble.json")
    args = ap.parse_args()

    out = {}
    for proto, groups in PROTOCOLS.items():
        _, res = evaluate(groups)
        out[proto] = {"groups": groups, **res}
        for tag, m in res.items():
            print(f"{proto:9s} {tag:10s} n={m['n']} Acc={m['acc'] * 100:.2f}% F1m={m['f1_macro'] * 100:.2f}% "
                  f"QWK={m['qwk']:.3f} ECE={m['ece']:.3f} sens={np.round(np.array(m['sensitivity']) * 100, 1)}")
    json.dump(out, open(args.out, "w"), indent=2)

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array(out["combined"]["mix_50_50"]["confusion"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True) * 100
        plt.figure(figsize=(6, 5))
        sns.heatmap(cmn, annot=np.vectorize(lambda v: f"{v:.2f}%")(cmn), fmt="", cmap="Greens", vmin=0, vmax=100,
                    annot_kws={"size": 14})
        plt.title("Normalized Predictions (%)", fontsize=14)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(args.fig, dpi=150)
        print("figura salva em", args.fig)


if __name__ == "__main__":
    main()
