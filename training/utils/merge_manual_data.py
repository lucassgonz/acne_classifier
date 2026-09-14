"""
Divide as imagens anotadas manualmente (data/manual_annotations/) em
treino/val/teste (70/15/15, por imagem, estratificado por classe) e funde
os crops de regiao correspondentes (data/manual_crops/) com o data/final/
existente (ACNE04), sem sobrescrever nada.

Uso:
    cd training
    python utils/merge_manual_data.py \
        --annotations-dir ../data/manual_annotations \
        --crops-dir ../data/manual_crops \
        --final-dir ../data/final
"""
import argparse
import os
import random
import shutil

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]


def split_images(annotations_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for cls in ["0", "1", "2", "3"]:
        cls_dir = os.path.join(annotations_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        base_names = [os.path.splitext(f)[0] for f in files]
        rng.shuffle(base_names)

        n = len(base_names)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)

        splits["val"].extend((b, cls) for b in base_names[:n_val])
        splits["test"].extend((b, cls) for b in base_names[n_val:n_val + n_test])
        splits["train"].extend((b, cls) for b in base_names[n_val + n_test:])

    return splits


def clear_previous_manual_crops(final_dir):
    """Remove todo crop manual_* de final_dir antes de mesclar de novo.

    Sem isso, rodar este script mais de uma vez (ex: apos anotar uma nova
    leva de imagens) recalcula o split 70/15/15 em cima de um pool maior,
    o que pode mover uma imagem de split -- mas a copia antiga, de um split
    diferente, nunca era removida, entao a MESMA imagem acabava presente em
    treino e teste ao mesmo tempo (vazamento de dado real, ja detectado).
    """
    removed = 0
    for region in REGIONS:
        for split in ["train", "val", "test"]:
            for cls in ["0", "1", "2", "3"]:
                d = os.path.join(final_dir, region, split, cls)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if fn.startswith("manual_"):
                        os.remove(os.path.join(d, fn))
                        removed += 1
    return removed


def merge_crops(splits, crops_dir, final_dir):
    stats = {"train": 0, "val": 0, "test": 0}
    missing_region_crops = 0

    for split_name, items in splits.items():
        for base_name, cls in items:
            for region in REGIONS:
                src = os.path.join(crops_dir, region, cls, f"{base_name}.jpg")
                if not os.path.exists(src):
                    missing_region_crops += 1
                    continue
                dest_dir = os.path.join(final_dir, region, split_name, cls)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f"manual_{base_name}.jpg")
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
                    stats[split_name] += 1

    return stats, missing_region_crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", default="../data/manual_annotations")
    parser.add_argument("--crops-dir", default="../data/manual_crops")
    parser.add_argument("--final-dir", default="../data/final")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Dividindo imagens em treino/val/teste...")
    splits = split_images(args.annotations_dir, seed=args.seed)
    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} imagens")

    removed = clear_previous_manual_crops(args.final_dir)
    print(f"\nCrops manuais antigos removidos antes de mesclar de novo: {removed}")

    print("Fundindo crops com data/final/...")
    stats, missing = merge_crops(splits, args.crops_dir, args.final_dir)

    print("\n" + "=" * 50)
    print("FUSAO CONCLUIDA")
    print("=" * 50)
    for split_name, n in stats.items():
        print(f"  {split_name}: {n} crops adicionados")
    print(f"  Total: {sum(stats.values())} crops")
    print(f"  Crops de regiao ausentes (imagem sem essa regiao detectada): {missing}")


if __name__ == "__main__":
    main()
