"""
Remonta data/final/{regiao}/{split}/{classe}/ usando os crops do YOLOv8 ja
gerados anteriormente em data/crops/, filtrando apenas os crops originarios
do ACNE04 (sem prefixo Train_/Validation_, que indicava Acne Level) e
atribuindo o split conforme o novo split por imagem (data/acne04_images/).

Uso:
    cd training
    python utils/rebuild_final_from_crops.py \
        --crops-dir ../data/crops \
        --split-dir ../data/acne04_images \
        --output ../data/final
"""
import argparse
import os
import re
import shutil

ACNE04_PATTERN = re.compile(r"^levle(\d+)_(\d+)\.jpg$", re.I)

REGION_FOLDER_MAP = {
    "forehead": "forehead",
    "chin": "chin",
    "nose": "nose",
    "left_chunk": "left_cheek",
    "right_chunk": "right_cheek",
}


def build_split_lookup(split_dir):
    """Retorna {filename: (split, classe)} a partir de data/acne04_images/{split}/{classe}/."""
    lookup = {}
    for split in ("train", "val", "test"):
        split_path = os.path.join(split_dir, split)
        if not os.path.isdir(split_path):
            continue
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fn in os.listdir(cls_path):
                lookup[fn] = (split, cls)
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default="../data/crops")
    parser.add_argument("--split-dir", default="../data/acne04_images")
    parser.add_argument("--output", default="../data/final")
    args = parser.parse_args()

    print("Lendo split ACNE04...")
    lookup = build_split_lookup(args.split_dir)
    print(f"  {len(lookup)} imagens mapeadas para split/classe")

    if os.path.isdir(args.output):
        print(f"Removendo data/final antigo ({args.output})...")
        shutil.rmtree(args.output)

    stats = {"train": 0, "val": 0, "test": 0}
    skipped_not_acne04 = 0
    skipped_no_split = 0

    for crop_folder, region in REGION_FOLDER_MAP.items():
        crop_dir = os.path.join(args.crops_dir, crop_folder)
        if not os.path.isdir(crop_dir):
            print(f"[!] Pasta não encontrada: {crop_dir}")
            continue

        for fn in os.listdir(crop_dir):
            if not ACNE04_PATTERN.match(fn):
                skipped_not_acne04 += 1
                continue

            if fn not in lookup:
                skipped_no_split += 1
                continue

            split, cls = lookup[fn]
            dest_dir = os.path.join(args.output, region, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.join(crop_dir, fn), os.path.join(dest_dir, fn))
            stats[split] += 1

    print("\n" + "=" * 50)
    print("data/final RECONSTRUÍDO (somente ACNE04)")
    print("=" * 50)
    for split, n in stats.items():
        print(f"  {split}: {n} crops")
    print(f"  Total: {sum(stats.values())} crops")
    print(f"\n  Crops do Acne Level ignorados: {skipped_not_acne04}")
    print(f"  Crops ACNE04 sem correspondência no split: {skipped_no_split}")


if __name__ == "__main__":
    main()
