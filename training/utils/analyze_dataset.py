"""
Análise do dataset combinado: sobreposição entre ACNE04 e Acne Level,
contagem de imagens únicas, crops por split e por região.

Uso:
    cd training
    python utils/analyze_dataset.py --acne04 ../../acne_1024 --acnelevel ../../Dataset --data-final ../../data/final
"""
import argparse
import json
import os
from collections import defaultdict


def load_acne04_filenames(acne04_dir):
    meta = os.path.join(acne04_dir, "metadata.jsonl")
    names = set()
    with open(meta) as f:
        for line in f:
            names.add(json.loads(line.strip())["file_name"])
    return names


def load_acnelevel_filenames(acnelevel_dir):
    names = set()
    for split in ["Train", "Validation"]:
        split_path = os.path.join(acnelevel_dir, split)
        if not os.path.isdir(split_path):
            continue
        for level_dir in os.listdir(split_path):
            level_path = os.path.join(split_path, level_dir)
            if os.path.isdir(level_path):
                for fn in os.listdir(level_path):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        names.add(fn)
    return names


def count_crops(data_final_dir):
    REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
    SPLITS = ["train", "val", "test"]
    stats = defaultdict(lambda: defaultdict(int))
    region_stats = defaultdict(int)

    for region in REGIONS:
        for split in SPLITS:
            for cls in range(4):
                d = os.path.join(data_final_dir, region, split, str(cls))
                if os.path.isdir(d):
                    n = sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
                    stats[split][region] += n
                    stats[split]["total"] += n
                    region_stats[region] += n

    return stats, region_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acne04", default="../../acne_1024")
    parser.add_argument("--acnelevel", default="../../Dataset")
    parser.add_argument("--data-final", default="../../data/final")
    args = parser.parse_args()

    print("=" * 60)
    print("ANÁLISE DE SOBREPOSIÇÃO ENTRE DATASETS")
    print("=" * 60)

    acne04 = load_acne04_filenames(args.acne04)
    acnelevel = load_acnelevel_filenames(args.acnelevel)
    overlap = acne04 & acnelevel

    print(f"ACNE04:      {len(acne04):,} imagens")
    print(f"Acne Level:  {len(acnelevel):,} imagens")
    print(f"Em comum:    {len(overlap):,} ({100*len(overlap)/len(acne04):.1f}% do ACNE04)")
    print(f"Únicas ACNE04: {len(acne04 - acnelevel):,}")
    print(f"Únicas Acne Level: {len(acnelevel - acne04):,}")
    print(f"Total ÚNICO (union): {len(acne04 | acnelevel):,}")
    print()

    if not os.path.isdir(args.data_final):
        print(f"[!] data/final não encontrado em {args.data_final}")
        return

    print("=" * 60)
    print("CONTAGEM DE CROPS EM data/final/")
    print("=" * 60)

    stats, region_stats = count_crops(args.data_final)
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}: {stats[split]['total']:,} crops")
        for region in ["forehead", "chin", "nose", "left_cheek", "right_cheek"]:
            print(f"  {region:<15} {stats[split][region]:>6}")

    print(f"\nTotal geral: {sum(v['total'] for v in stats.values()):,} crops")
    print()
    print("=" * 60)
    print("ESCLARECIMENTO DA UNIDADE AMOSTRAL")
    print("=" * 60)
    total_unique_images = len(acne04 | acnelevel)
    total_crops = sum(v["total"] for v in stats.values())
    test_crops = stats["test"]["total"]
    avg_crops = total_crops / total_unique_images if total_unique_images else 0

    print(f"Imagens únicas nos datasets originais: {total_unique_images:,}")
    print(f"Total de crops gerados (todas as regiões, todos os splits): {total_crops:,}")
    print(f"Média de crops por imagem: {avg_crops:.2f}")
    print(f"Crops no conjunto de teste: {test_crops:,}")
    print()
    print("NOTA: A unidade amostral do classificador é o CROP REGIONAL (224×224),")
    print("não a imagem facial original. O split train/val/test foi aplicado ao")
    print("nível da imagem original antes da segmentação YOLOv8, garantindo que")
    print("regiões de uma mesma face não apareçam em splits distintos.")


if __name__ == "__main__":
    main()
