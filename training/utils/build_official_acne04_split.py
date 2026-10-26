"""
Usa o split oficial de 5-fold do ACNE04 (Wu et al., ICCV 2019, repo
github.com/xpwu95/LDL), baixado do Google Drive oficial dos autores
(Detection.tar -> VOC2007/ImageSets/Main/NNEW_trainval_X.txt +
NNEW_test_X.txt), pra permitir comparacao DIRETA (mesmo split) com JAGC
(84,11% acc) e trabalhos que reusaram esse split.

Confirmado: nosso `Dataset/` (pasta "AcneLevel") tem os MESMOS 1457
arquivos (nome a nome) do release oficial -- nao precisamos baixar as
imagens deles (Classification.tar, 1GB), so os indices de split + os
rotulos oficiais (que usamos aqui no lugar do rotulo da pasta Level X --
2,9% dos arquivos tem grau diferente entre a pasta local e o oficial,
nossa pasta local tende a ser 1 grau mais severa em casos de fronteira).

Cada NNEW_trainval_X.txt vira treino+val (90/10 estratificado, sem grupo
-- fonte unica, sem risco de vazamento cross-source) e NNEW_test_X.txt
vira teste, fold a fold.

Uso:
    cd training
    python utils/build_official_acne04_split.py
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold


def index_dataset_files(dataset_root):
    """Mapeia nome de arquivo -> caminho, procurando em todas as pastas
    Level X (o rotulo oficial pode diferir do rotulo da pasta local)."""
    index = {}
    for p in glob.glob(os.path.join(dataset_root, "Train", "Level *", "*.jpg")) + \
             glob.glob(os.path.join(dataset_root, "Validation", "Level *", "*.jpg")):
        index[os.path.basename(p)] = p
    return index


def read_split_file(path):
    items = []
    for line in open(path):
        name, label, lesion_count = line.split()
        items.append((name, int(label), int(lesion_count)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True,
                     help="Caminho pra VOC2007/ImageSets/Main extraido do Detection.tar oficial")
    ap.add_argument("--dataset-root", default="../Dataset")
    ap.add_argument("--output", default="results/kfold_splits_official_acne04.json")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    file_index = index_dataset_files(args.dataset_root)
    print(f"Imagens locais indexadas: {len(file_index)}")

    folds = []
    mismatches_total, not_found_total = 0, 0
    for fold in range(5):
        trainval = read_split_file(os.path.join(args.official_dir, f"NNEW_trainval_{fold}.txt"))
        test = read_split_file(os.path.join(args.official_dir, f"NNEW_test_{fold}.txt"))

        def resolve(items):
            names, labels, lesions, missing = [], [], [], 0
            for name, label, lesion_count in items:
                if name not in file_index:
                    missing += 1
                    continue
                names.append(name)
                labels.append(label)
                lesions.append(lesion_count)
            return names, labels, lesions, missing

        tv_names, tv_labels, tv_lesions, tv_missing = resolve(trainval)
        test_names, test_labels, test_lesions, test_missing = resolve(test)
        not_found_total += tv_missing + test_missing

        tv_names, tv_labels = np.array(tv_names), np.array(tv_labels)
        skf = StratifiedKFold(n_splits=int(round(1 / args.val_frac)), shuffle=True, random_state=args.seed + fold)
        train_idx, val_idx = next(skf.split(tv_names, tv_labels))

        folds.append({
            "fold": fold,
            "train": tv_names[train_idx].tolist(),
            "val": tv_names[val_idx].tolist(),
            "test": test_names,
            "train_labels_official": {n: int(l) for n, l in zip(tv_names[train_idx], tv_labels[train_idx])},
            "val_labels_official": {n: int(l) for n, l in zip(tv_names[val_idx], tv_labels[val_idx])},
            "test_labels_official": {n: int(l) for n, l in zip(test_names, test_labels)},
            "test_lesion_counts": {n: int(c) for n, c in zip(test_names, test_lesions)},
        })
        print(f"fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_names)} "
              f"(faltando localmente: {tv_missing + test_missing})")

    with open(args.output, "w") as f:
        json.dump({"k": 5, "seed": args.seed, "source": "NNEW split oficial (xpwu95/LDL, ICCV2019)",
                   "folds": folds}, f)
    print(f"Salvo em {args.output}")
    print(f"Total de imagens do split oficial nao encontradas localmente: {not_found_total}")


if __name__ == "__main__":
    main()
