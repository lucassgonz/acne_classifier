"""
Split train/val/test (70/15/15) usando apenas o ACNE04, ao nível da
imagem original (antes da segmentação YOLOv8), garantindo que todos os
crops regionais de uma mesma imagem fiquem no mesmo split.

Uso:
    cd training
    python utils/prepare_acne04_split.py \
        --acne04 ../../acne_1024 \
        --output ../../data/acne04_images \
        --seed 42
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict

from tqdm import tqdm


def parse_acne04(acne04_dir):
    samples = []
    with open(os.path.join(acne04_dir, "metadata.jsonl")) as f:
        for line in f:
            fn = json.loads(line)["file_name"]
            m = re.match(r"levle(\d+)_(\d+)", fn, re.I)
            if m:
                label = int(m.group(1))
                samples.append({
                    "path": os.path.join(acne04_dir, fn),
                    "label": label,
                    "filename": fn,
                })
    return samples


def split_images(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(s)

    splits = {"train": [], "val": [], "test": []}
    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        splits["val"].extend(items[:n_val])
        splits["test"].extend(items[n_val:n_val + n_test])
        splits["train"].extend(items[n_val + n_test:])

    return splits


def copy_splits(splits, output_dir):
    for split_name, samples in splits.items():
        for s in tqdm(samples, desc=split_name):
            dest_dir = os.path.join(output_dir, split_name, str(s["label"]))
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, s["filename"])
            if not os.path.exists(dest):
                shutil.copy2(s["path"], dest)


def print_stats(splits):
    print("\n" + "=" * 50)
    print("SPLIT ACNE04 (imagem-level, 70/15/15)")
    print("=" * 50)
    total = sum(len(v) for v in splits.values())
    for split_name in ["train", "val", "test"]:
        by_class = defaultdict(int)
        for s in splits[split_name]:
            by_class[s["label"]] += 1
        print(f"\n{split_name.upper()}: {len(splits[split_name])} imagens")
        for cls in sorted(by_class):
            print(f"  Level {cls}: {by_class[cls]}")
    print(f"\nTotal: {total} imagens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acne04", default="../../acne_1024")
    parser.add_argument("--output", default="../../data/acne04_images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Carregando ACNE04...")
    samples = parse_acne04(args.acne04)
    print(f"  {len(samples)} imagens")

    splits = split_images(samples, seed=args.seed)
    print_stats(splits)

    print(f"\nCopiando arquivos para {args.output}...")
    copy_splits(splits, args.output)
    print("Concluído.")
    print("\nPróximo passo: rodar a segmentação YOLOv8 sobre")
    print(f"  {args.output}/train, {args.output}/val, {args.output}/test")
    print("preservando os subdiretórios de split, para gerar os crops em")
    print("data/final/{regiao}/{split}/{classe}/.")


if __name__ == "__main__":
    main()
