"""
Correcao de viés por classe nas probabilidades de um ensemble, com validacao cruzada ENTRE folds:
os offsets (em log-prob) do fold k sao ajustados so com as predicoes dos outros 4 folds -- que
vem de modelos que nao viram essas imagens -- e aplicados ao fold k. Nada e ajustado no teste.

Uso: python analyze_bias_correction.py --dirs results/kfold_wholeimage/jagc_replica_resnet18 \
         results/kfold_wholeimage/jagc_replica_resnet18_seed43 ... [--key probs|probs_tta]
"""
import argparse
import glob
import os

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

GRID = np.linspace(-1.5, 1.5, 61)


def score(logp, y, b):
    pr = (logp + b).argmax(1)
    return 0.5 * (accuracy_score(y, pr) + f1_score(y, pr, average="macro"))


def fit_offsets(logp, y, passes=3):
    b = np.zeros(logp.shape[1])
    for _ in range(passes):
        for c in range(1, logp.shape[1]):  # classe 0 fixa como referencia
            best = max(GRID, key=lambda v: score(logp, y, np.where(np.arange(len(b)) == c, v, b)))
            b[c] = best
    return b


def metrics(y, pr):
    return (accuracy_score(y, pr) * 100, f1_score(y, pr, average="macro") * 100, cohen_kappa_score(y, pr, weights="quadratic"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--key", default="probs")
    args = ap.parse_args()

    per_fold = []
    for k in range(5):
        zs = [np.load(os.path.join(d, f"fold{k}.npz"), allow_pickle=True) for d in args.dirs]
        for z in zs[1:]:
            assert (z["names"] == zs[0]["names"]).all() and (z["labels"] == zs[0]["labels"]).all()
        per_fold.append((np.mean([z[args.key] for z in zs], axis=0), zs[0]["labels"]))

    logp = [np.log(p + 1e-9) for p, _ in per_fold]
    ys = [y for _, y in per_fold]
    base_pred, corr_pred = [], []
    print(f"ensemble de {len(args.dirs)} dirs, chave '{args.key}'")
    for k in range(5):
        tr = [i for i in range(5) if i != k]
        b = fit_offsets(np.concatenate([logp[i] for i in tr]), np.concatenate([ys[i] for i in tr]))
        base_pred.append(logp[k].argmax(1))
        corr_pred.append((logp[k] + b).argmax(1))
        print(f"  fold {k}: offsets={np.round(b, 2)}  acc {accuracy_score(ys[k], base_pred[-1])*100:.2f} -> {accuracy_score(ys[k], corr_pred[-1])*100:.2f}")
    Y = np.concatenate(ys)
    for tag, pr in (("sem correcao", np.concatenate(base_pred)), ("com correcao (cross-fit)", np.concatenate(corr_pred))):
        a, f, q = metrics(Y, pr)
        cm = confusion_matrix(Y, pr)
        print(f"{tag:26s} n={len(Y)} Acc={a:.2f}% F1m={f:.2f}% QWK={q:.3f} | sens por classe={np.round(cm.diagonal()/cm.sum(1)*100, 1)}")


if __name__ == "__main__":
    main()
