"""
Gera o split train/val/test por SUJEITO, usando todas as 2.863 imagens
dos dois datasets (ACNE04 + Acne Level) sem data leakage.

Identificador de sujeito: par (level, número) extraído do nome do arquivo.
Ex: levle0_14.jpg → sujeito (0, 14).
Sujeitos com fotos nos dois datasets têm AMBAS as fotos no mesmo split.

Estrutura de saída:
    data/subject_split/
        train/  val/  test/
            0/  1/  2/  3/
                levle0_14_acne04.jpg   ← renomeado para evitar colisão
                levle0_14_acnelevel.jpg

Uso:
    cd training
    python utils/prepare_subject_split.py \
        --acne04 ../../acne_1024 \
        --acnelevel ../../Dataset \
        --output ../../data/subject_split \
        --seed 42
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_acne04(acne04_dir):
    """Retorna lista de (caminho, label, subject_key) para o ACNE04."""
    samples = []
    with open(os.path.join(acne04_dir, "metadata.jsonl")) as f:
        for line in f:
            fn = json.loads(line)["file_name"]
            m = re.match(r"levle(\d+)_(\d+)", fn, re.I)
            if m:
                label, num = int(m.group(1)), int(m.group(2))
                samples.append({
                    "path": os.path.join(acne04_dir, fn),
                    "label": label,
                    "subject": (label, num),
                    "source": "acne04",
                    "filename": fn,
                })
    return samples


def parse_acnelevel(acnelevel_dir):
    """Retorna lista de (caminho, label, subject_key) para o Acne Level."""
    samples = []
    for split_folder in ["Train", "Validation"]:
        split_path = os.path.join(acnelevel_dir, split_folder)
        if not os.path.isdir(split_path):
            continue
        for level_dir in os.listdir(split_path):
            level_path = os.path.join(split_path, level_dir)
            if not os.path.isdir(level_path):
                continue
            label = int(level_dir.replace("Level", "").strip())
            for fn in os.listdir(level_path):
                m = re.match(r"levle(\d+)_(\d+)", fn, re.I)
                if m:
                    num = int(m.group(2))
                    samples.append({
                        "path": os.path.join(level_path, fn),
                        "label": label,
                        "subject": (label, num),
                        "source": "acnelevel",
                        "filename": fn,
                    })
    return samples


def split_subjects(all_samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Agrupa por sujeito e divide os SUJEITOS em train/val/test.
    Todas as fotos de um sujeito vão para o mesmo split.
    """
    by_subject = defaultdict(list)
    for s in all_samples:
        by_subject[s["subject"]].append(s)

    subjects = sorted(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val_subjects = set(subjects[:n_val])
    test_subjects = set(subjects[n_val:n_val + n_test])
    train_subjects = set(subjects[n_val + n_test:])

    splits = {"train": [], "val": [], "test": []}
    for subj, photos in by_subject.items():
        if subj in train_subjects:
            splits["train"].extend(photos)
        elif subj in val_subjects:
            splits["val"].extend(photos)
        else:
            splits["test"].extend(photos)

    return splits, {"train": train_subjects, "val": val_subjects, "test": test_subjects}


def copy_splits(splits, output_dir):
    """Copia os arquivos para a estrutura de saída, evitando colisão de nomes."""
    for split_name, samples in splits.items():
        for s in tqdm(samples, desc=split_name):
            label = s["label"]
            source = s["source"]
            base, ext = os.path.splitext(s["filename"])
            # Adiciona sufixo de fonte para evitar colisão entre datasets
            new_name = f"{base}_{source}{ext}"
            dest_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, new_name)
            if not os.path.exists(dest):
                shutil.copy2(s["path"], dest)


def print_stats(splits, subject_splits):
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DO SPLIT POR SUJEITO")
    print("=" * 60)

    total_photos = sum(len(v) for v in splits.values())
    total_subjects = sum(len(v) for v in subject_splits.values())

    for split_name in ["train", "val", "test"]:
        photos = splits[split_name]
        subjs = subject_splits[split_name]
        acne04 = sum(1 for p in photos if p["source"] == "acne04")
        acnelevel = sum(1 for p in photos if p["source"] == "acnelevel")
        by_class = defaultdict(int)
        for p in photos:
            by_class[p["label"]] += 1
        print(f"\n{split_name.upper()}: {len(photos)} fotos, {len(subjs)} sujeitos")
        print(f"  ACNE04: {acne04}  |  Acne Level: {acnelevel}")
        for cls in sorted(by_class):
            print(f"  Level {cls}: {by_class[cls]}")

    print(f"\nTOTAL: {total_photos} fotos, {total_subjects} sujeitos únicos")
    print("(Reportar no paper: 2.863 imagens de 1.457 sujeitos)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acne04", default="../../acne_1024")
    parser.add_argument("--acnelevel", default="../../Dataset")
    parser.add_argument("--output", default="../../data/subject_split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Carregando ACNE04...")
    acne04 = parse_acne04(args.acne04)
    print(f"  {len(acne04)} imagens")

    print("Carregando Acne Level...")
    acnelevel = parse_acnelevel(args.acnelevel)
    print(f"  {len(acnelevel)} imagens")

    all_samples = acne04 + acnelevel
    print(f"\nTotal combinado: {len(all_samples)} imagens")

    print("Dividindo por sujeito (70/15/15)...")
    splits, subject_splits = split_subjects(all_samples, seed=args.seed)

    print_stats(splits, subject_splits)

    print(f"\nCopiando arquivos para {args.output}...")
    copy_splits(splits, args.output)
    print("Concluído.")


if __name__ == "__main__":
    main()
