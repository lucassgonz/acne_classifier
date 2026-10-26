"""
Cria data/final_v3_production/{regiao}/{train,val,test}/{classe}/ (via
symlink, sem duplicar arquivo) a partir de data/final_v3/{regiao}/train/,
com um split FIXO (nao k-fold) agrupado por identidade canonica --
usado para treinar o ensemble de PRODUCAO (deploy no app), que deve usar
o MAXIMO de dado possivel pra treino, ja que a generalizacao real ja foi
validada honestamente pelo k-fold (results/kfold_splits_v3.json).

90% train / 7% val (early stopping) / 3% test (so pra o script nao
quebrar, numero nao usado pra nada alem de log).

Uso:
    cd training
    python utils/build_production_split.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]


def main():
    manifest = json.load(open("../data/final_v3/manifest.json"))
    names = np.array([m["name"] for m in manifest])
    y = np.array([m["level"] for m in manifest])
    canonical = [m["canonical"] for m in manifest]
    ids = {c: i for i, c in enumerate(sorted(set(canonical)))}
    groups = np.array([ids[c] for c in canonical])
    regions_by_name = {m["name"]: m["regions"] for m in manifest}

    # ~90/10 grouped split (train+val) via 10-fold, pega 1 fold pro val/test
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=123)
    trainval_idx, holdout_idx = next(skf.split(names, y, groups))
    # do holdout (10%), metade val metade test
    holdout_groups = groups[holdout_idx]
    skf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=123)
    val_rel, test_rel = next(skf2.split(names[holdout_idx], y[holdout_idx], holdout_groups))
    val_idx, test_idx = holdout_idx[val_rel], holdout_idx[test_rel]

    assert not (set(groups[trainval_idx]) & set(groups[val_idx]))
    assert not (set(groups[trainval_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))

    split_of = {}
    for i in trainval_idx:
        split_of[names[i]] = "train"
    for i in val_idx:
        split_of[names[i]] = "val"
    for i in test_idx:
        split_of[names[i]] = "test"

    out_root = "../data/final_v3_production"
    counts = defaultdict(int)
    for m in manifest:
        name, level, regions = m["name"], m["level"], m["regions"]
        split = split_of[name]
        for region in regions:
            src = os.path.abspath(f"../data/final_v3/{region}/train/{level}/{name}")
            dst_dir = f"{out_root}/{region}/{split}/{level}"
            os.makedirs(dst_dir, exist_ok=True)
            dst = f"{dst_dir}/{name}"
            if not os.path.exists(dst):
                os.symlink(src, dst)
            counts[(region, split)] += 1

    print(f"Imagens: {len(names)} | grupos: {len(set(groups))} "
          f"| train={len(trainval_idx)} val={len(val_idx)} test={len(test_idx)}")
    for region in REGIONS:
        print(region, {s: counts[(region, s)] for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
