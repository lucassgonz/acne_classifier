"""
Ensemble final: media das probabilidades softmax das 5 seeds (42-46) de
uma config de imagem inteira (default: whole_base + SWA + TTA), por
imagem -- a mesma receita do modelo final region-aware do artigo.

Cada seed ja tem TTA (7 views) aplicado na propria predicao salva; aqui so
fazemos a segunda camada de ensemble (entre seeds), nao entre views.

Uso:
    cd training
    python analyze_wholeimage_ensemble.py
    python analyze_wholeimage_ensemble.py --config whole_v2_cbam_swa_tta
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, accuracy_score

GRADES = ["0-clear", "1-mild", "2-moderate", "3-severe"]
SEEDS = [42, 43, 44, 45, 46]
BASE_CONFIG = "whole_base_swa_tta"


def config_dir(seed):
    name = BASE_CONFIG if seed == 42 else f"{BASE_CONFIG}_seed{seed}"
    return Path("results/kfold_wholeimage") / name


def load_seed(seed, fold):
    d = np.load(config_dir(seed) / f"fold{fold}.npz", allow_pickle=True)
    return d["names"], d["labels"], d["probs"]


def sensitivity_specificity_per_class(y_true, y_pred, n_classes):
    out = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        out.append({"grade": GRADES[c], "sensitivity": sens, "specificity": spec, "support": int(tp + fn)})
    return out


def expected_calibration_error(y_true, probs, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def main():
    global BASE_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=BASE_CONFIG)
    args = ap.parse_args()
    BASE_CONFIG = args.config

    all_labels, all_ens_probs = [], []
    for fold in range(5):
        names0, labels0, _ = load_seed(SEEDS[0], fold)
        fold_probs = []
        for seed in SEEDS:
            names, labels, probs = load_seed(seed, fold)
            assert list(names) == list(names0), f"order mismatch seed {seed} fold {fold}"
            assert list(labels) == list(labels0)
            fold_probs.append(probs)
        ensemble_probs = np.mean(fold_probs, axis=0)
        all_labels.append(labels0)
        all_ens_probs.append(ensemble_probs)

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_ens_probs)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    per_class = sensitivity_specificity_per_class(labels, preds, 4)
    ece = expected_calibration_error(labels, probs, n_bins=10)

    report = {
        "config": f"{BASE_CONFIG}_ensemble5seeds", "seeds": SEEDS, "n_samples_pooled": int(len(labels)),
        "accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted),
        "qwk": float(qwk), "confusion_matrix": cm.tolist(), "confusion_matrix_labels": GRADES,
        "per_grade": per_class, "calibration": {"expected_calibration_error": ece},
    }
    out_path = Path(f"results/full_metrics_wholeimage_{BASE_CONFIG}_ensemble5seeds.json")
    out_path.write_text(json.dumps(report, indent=2))

    print(f"=== {BASE_CONFIG} ENSEMBLE (5 seeds x 5 folds, n={len(labels)}) ===")
    print(f"Accuracy:     {acc*100:.2f}%")
    print(f"Macro-F1:     {f1_macro*100:.2f}%")
    print(f"Weighted-F1:  {f1_weighted*100:.2f}%")
    print(f"QWK:          {qwk:.3f}")
    print(f"ECE:          {ece:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=pred), order:", GRADES)
    print(cm)
    print()
    print("Per-grade sensitivity / specificity:")
    for row in per_class:
        print(f"  {row['grade']:>12}: sens={row['sensitivity']*100:5.1f}%  spec={row['specificity']*100:5.1f}%  n={row['support']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
