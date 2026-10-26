"""
Resume as execucoes de run_kfold_clean.py: metricas por fold (media +- desvio),
metricas agregadas fora-da-amostra e comparacao pareada contra uma
configuracao de referencia (diferenca por fold e teste de McNemar sobre as
predicoes agregadas).

Uso:
    cd training
    python summarize_kfold.py --baseline adam_base
"""
import argparse
import glob
import os

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def load(name, root):
    folds = {}
    for path in sorted(glob.glob(os.path.join(root, name, "fold*.npz"))):
        k = int(os.path.basename(path)[4:-4])
        folds[k] = dict(np.load(path))
    return folds


def pooled(folds):
    keys = sorted(folds)
    names = np.concatenate([folds[k]["names"] for k in keys])
    regions = np.concatenate([folds[k]["regions"] for k in keys])
    labels = np.concatenate([folds[k]["labels"] for k in keys])
    probs = np.concatenate([folds[k]["probs"] for k in keys])
    return names, regions, labels, probs


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(1 - stats.chi2.cdf(stat, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/kfold")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()

    names = args.configs or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    data = {n: load(n, args.root) for n in names}
    data = {n: f for n, f in data.items() if f}

    print(f"{'config':<26}{'folds':>6}{'acc (media±dp)':>18}{'F1':>8}{'QWK':>8}{'acc agreg.':>12}{'min/fold':>10}")
    for n, folds in data.items():
        m = np.array([folds[k]["metrics"] for k in sorted(folds)])
        _, _, y, p = pooled(folds)
        mins = np.mean([float(folds[k]["minutes"][0]) for k in folds])
        print(f"{n:<26}{len(folds):>6}{m[:, 0].mean()*100:>12.2f}±{m[:, 0].std()*100:<4.2f}"
              f"{m[:, 1].mean():>8.3f}{m[:, 2].mean():>8.3f}{accuracy_score(y, p.argmax(1))*100:>11.2f}%{mins:>10.1f}")

    if args.baseline and args.baseline in data:
        base = data[args.baseline]
        print(f"\nComparacao pareada contra '{args.baseline}' (apenas folds em comum):")
        print(f"{'config':<26}{'dAcc/fold':>12}{'folds+':>8}{'dQWK':>9}{'McNemar b/c':>14}{'p':>8}")
        for n, folds in data.items():
            if n == args.baseline:
                continue
            common = sorted(set(folds) & set(base))
            if len(common) < 2:
                continue
            da = [folds[k]["metrics"][0] - base[k]["metrics"][0] for k in common]
            dq = [folds[k]["metrics"][2] - base[k]["metrics"][2] for k in common]
            fa = {k: folds[k] for k in common}
            fb = {k: base[k] for k in common}
            _, _, ya, pa = pooled(fa)
            _, _, _, pb = pooled(fb)
            b, c, pv = mcnemar(pa.argmax(1) == ya, pb.argmax(1) == ya)
            print(f"{n:<26}{np.mean(da)*100:>+11.2f}pp{sum(d > 0 for d in da):>5}/{len(da):<2}"
                  f"{np.mean(dq):>+9.3f}{f'{b}/{c}':>14}{pv:>8.3f}")


if __name__ == "__main__":
    main()
