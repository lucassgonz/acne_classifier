"""
Adiciona contagem REAL de licoes (do split oficial NNEW, Wu et al.
ICCV2019) a train/val tambem (so test tinha, ver
build_official_acne04_split.py) -- usado pro treino multi-task
severidade+contagem (replica a supervisao conjunta do JAGC).

~114/1457 imagens (todas grau 3) tem contagem inconsistente entre os
arquivos NNEW_trainval_X/test_X (provavel ruido na anotacao de licoes
densas/sobrepostas em casos muito severos) -- usamos a MEDIA dos valores
observados pra essas, documentado aqui honestamente.

Uso:
    cd training
    python utils/add_lesion_counts_to_official_split.py
"""
import glob
import json
from collections import defaultdict

import numpy as np

OFFICIAL_DIR = ("/private/tmp/claude-501/-Users-GonLu-Documents-Projetos-pibic-final/"
                 "927f7222-5870-4ba0-a727-5729557b32fe/scratchpad/acne04_official/"
                 "Detection_extracted/Detection/VOC2007/ImageSets/Main")


def main():
    counts = defaultdict(list)
    for f in glob.glob(f"{OFFICIAL_DIR}/NNEW_*.txt"):
        for line in open(f):
            name, label, count = line.split()
            counts[name].append(int(count))

    n_inconsistent = sum(1 for v in counts.values() if len(set(v)) > 1)
    mean_count = {name: float(np.mean(v)) for name, v in counts.items()}
    print(f"Imagens com contagem: {len(mean_count)} | inconsistentes entre arquivos: {n_inconsistent} "
          f"(usando media)")

    path = "results/kfold_splits_official_acne04.json"
    d = json.load(open(path))
    for fold in d["folds"]:
        for split in ("train", "val", "test"):
            key = f"{split}_lesion_counts"
            fold[key] = {name: mean_count[name] for name in fold[split] if name in mean_count}
            missing = [name for name in fold[split] if name not in mean_count]
            if missing:
                print(f"fold {fold['fold']} {split}: {len(missing)} sem contagem (nao deveria acontecer)")

    json.dump(d, open(path, "w"))
    print(f"Salvo em {path}")


if __name__ == "__main__":
    main()
