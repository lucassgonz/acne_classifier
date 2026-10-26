"""
Mesmo relatorio de analyze_full_metrics.py, mas para os resultados de
imagem inteira (results/kfold_wholeimage/<config>/fold*.npz), que nao tem
campo 'regions' (um rotulo por imagem, nao por recorte).

Uso:
    cd training
    python analyze_wholeimage_metrics.py --config whole_base
    python analyze_wholeimage_metrics.py --config whole_ordinal_ls
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, accuracy_score

GRADES = ["0-clear", "1-mild", "2-moderate", "3-severe"]


def load_pooled(config_dir: Path):
    names, labels, probs = [], [], []
    fold_files = sorted(config_dir.glob("fold*.npz"))
    if not fold_files:
        raise FileNotFoundError(f"No fold*.npz under {config_dir}")
    for f in fold_files:
        d = np.load(f, allow_pickle=True)
        names.append(d["names"])
        labels.append(d["labels"])
        probs.append(d["probs"])
    return np.concatenate(names), np.concatenate(labels), np.concatenate(probs), len(fold_files)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config_dir = Path("results/kfold_wholeimage") / args.config
    names, labels, probs, n_folds = load_pooled(config_dir)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    per_class = sensitivity_specificity_per_class(labels, preds, 4)
    ece = expected_calibration_error(labels, probs, n_bins=10)

    report = {
        "config": args.config, "n_folds": n_folds, "n_samples_pooled": int(len(labels)),
        "accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted),
        "qwk": float(qwk), "confusion_matrix": cm.tolist(), "confusion_matrix_labels": GRADES,
        "per_grade": per_class, "calibration": {"expected_calibration_error": ece},
    }
    out_path = Path("results") / f"full_metrics_wholeimage_{args.config}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"=== {args.config} (pooled across {n_folds} folds, n={len(labels)}) ===")
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
