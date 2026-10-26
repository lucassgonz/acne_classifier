"""
Gera kfold_splits_wholeimage_v2.json (k=5) para o manifest de imagem
inteira Acne04+AcneLevel(tratado) (build_wholeimage_manifest_v2.py),
agrupando por identidade CANONICA -- a versao Acne04 e a versao AcneLevel
tratada da MESMA foto sempre ficam no mesmo fold (mesmo raciocinio de
build_kfold_splits_v2.py, usado no pipeline region-crop).

Uso:
    cd training
    python utils/build_kfold_splits_wholeimage_v2.py
"""
import argparse
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../data/final_wholeimage_v2/manifest.json")
    ap.add_argument("--output", default="results/kfold_splits_wholeimage_v2.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-parts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    names = np.array([m["name"] for m in manifest])
    y = np.array([m["level"] for m in manifest])
    canonical = [m["canonical"] for m in manifest]
    ids = {c: i for i, c in enumerate(sorted(set(canonical)))}
    groups = np.array([ids[c] for c in canonical])

    print(f"Imagens: {len(names)} | grupos (fotos unicas): {len(set(groups))} | "
          f"maior grupo: {Counter(groups).most_common(1)[0][1]}")

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
        print(f"fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | classes teste={dist(test_idx)}")

    with open(args.output, "w") as f:
        json.dump({"k": args.k, "seed": args.seed, "folds": folds}, f)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
