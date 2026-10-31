"""
Mistura (media de probabilidades) entre dois grupos de modelos no dataset combinado, alinhada por nome de
imagem, com opcao de correcao de vies por classe validada entre folds (offsets de cada fold ajustados so
nos outros 4). Reporta geral e por fonte (a04_ / lvl_).

Uso: python analyze_mix_combined.py --a DIR [DIR ...] --b DIR [DIR ...] [--weight 0.5] [--bias] [--key probs]
"""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from analyze_bias_correction import fit_offsets


def load(d, k, key):
    z = np.load(f"{d}/fold{k}.npz", allow_pickle=True)
    return dict(zip(z["names"], z[key])), dict(zip(z["names"], z["labels"]))


def mean_probs(dirs, k, names, key):
    return np.mean([[load(d, k, key)[0][n] for n in names] for d in dirs], axis=0)


def report(tag, pred, Y, N):
    a = np.array([n.startswith("a04_") for n in N])
    sub = lambda m: f"{accuracy_score(Y[m], pred[m]) * 100:.2f}/{f1_score(Y[m], pred[m], average='macro') * 100:.2f}"
    print(f"{tag:30s} n={len(Y)} Acc={accuracy_score(Y, pred) * 100:.2f}% F1m={f1_score(Y, pred, average='macro') * 100:.2f}% "
          f"QWK={cohen_kappa_score(Y, pred, weights='quadratic'):.3f} | a04 {sub(a)} | lvl {sub(~a)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="grupo A (ex: receita do artigo)")
    ap.add_argument("--b", nargs="+", required=True, help="grupo B (ex: replica JAGC)")
    ap.add_argument("--weight", type=float, default=0.5, help="peso do grupo A")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--key", default="probs")
    args = ap.parse_args()

    folds = []
    for k in range(5):
        names = sorted(load(args.a[0], k, args.key)[0])
        y = np.array([load(args.a[0], k, args.key)[1][n] for n in names])
        folds.append((names, y, mean_probs(args.a, k, names, args.key), mean_probs(args.b, k, names, args.key)))
    Y = np.concatenate([f[1] for f in folds])
    N = np.concatenate([f[0] for f in folds])
    PA = np.concatenate([f[2] for f in folds])
    PB = np.concatenate([f[3] for f in folds])
    report("grupo A", PA.argmax(1), Y, N)
    report("grupo B", PB.argmax(1), Y, N)
    mix = [args.weight * f[2] + (1 - args.weight) * f[3] for f in folds]
    report(f"mistura (peso A={args.weight})", np.concatenate(mix).argmax(1), Y, N)
    if args.bias:
        lp = [np.log(m + 1e-9) for m in mix]
        pred = []
        for k in range(5):
            tr = [i for i in range(5) if i != k]
            b = fit_offsets(np.concatenate([lp[i] for i in tr]), np.concatenate([folds[i][1] for i in tr]))
            pred.append((lp[k] + b).argmax(1))
        report("mistura + vies (cross-fit)", np.concatenate(pred), Y, N)


if __name__ == "__main__":
    main()
