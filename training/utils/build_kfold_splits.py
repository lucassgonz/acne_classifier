"""
Gera folds (k=5) ao nivel de imagem, estratificados por classe e agrupados
por clusters de quase duplicatas (pHash, componentes conexas com distancia
Hamming <= limiar), de modo que copias e fotos muito parecidas nunca fiquem
em folds diferentes. Cada fold tem teste (1 fold), validacao (fatia do
restante, tambem por grupo) e treino.

Uso:
    cd training
    python utils/build_kfold_splits.py --images-dir ../data/wholeimage \
        --phash-report results/phash_report.json --output results/kfold_splits.json
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def collect(images_dir):
    items = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "*", "*", "*"))):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            parts = path.split(os.sep)
            items[parts[-1]] = int(parts[-2])
    return items


def cluster_groups(names, report_path, threshold):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with open(report_path) as f:
        pairs = json.load(f)["near_pairs"]
    for p in pairs:
        if p["distance"] <= threshold:
            a, b = os.path.basename(p["a"]), os.path.basename(p["b"])
            if a in parent and b in parent:
                parent[find(a)] = find(b)
    roots = {n: find(n) for n in names}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return np.array([ids[roots[n]] for n in names])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="../data/wholeimage")
    parser.add_argument("--phash-report", default="results/phash_report.json")
    parser.add_argument("--output", default="results/kfold_splits.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=6)
    parser.add_argument("--val-parts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = collect(args.images_dir)
    names = np.array(sorted(items))
    y = np.array([items[n] for n in names])
    groups = cluster_groups(names, args.phash_report, args.threshold)
    print(f"Imagens: {len(names)} | grupos: {len(set(groups))} | maior grupo: {Counter(groups).most_common(1)[0][1]}")

    outer = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    folds = []
    for fold, (train_val_idx, test_idx) in enumerate(outer.split(names, y, groups)):
        inner = StratifiedGroupKFold(n_splits=args.val_parts, shuffle=True, random_state=args.seed + fold)
        tr_rel, va_rel = next(inner.split(names[train_val_idx], y[train_val_idx], groups[train_val_idx]))
        train_idx, val_idx = train_val_idx[tr_rel], train_val_idx[va_rel]
        folds.append({
            "fold": fold,
            "train": names[train_idx].tolist(),
            "val": names[val_idx].tolist(),
            "test": names[test_idx].tolist(),
        })

        g_sets = [set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])]
        assert not (g_sets[0] & g_sets[1]) and not (g_sets[0] & g_sets[2]) and not (g_sets[1] & g_sets[2])
        dist = lambda idx: dict(sorted(Counter(y[idx]).items()))
        manual = lambda idx: int(sum(n.startswith("manual_") for n in names[idx]))
        print(f"fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | "
              f"classes teste={dist(test_idx)} | manual no teste={manual(test_idx)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"k": args.k, "threshold": args.threshold, "seed": args.seed, "folds": folds}, f)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
