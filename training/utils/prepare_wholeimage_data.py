"""
Reconstroi a mesma divisao treino/val/teste (por imagem) ja usada no
pipeline de recorte por regiao (data/final/), mas copiando as imagens
ORIGINAIS INTEIRAS (sem crop de regiao) para data/wholeimage/.

Objetivo: isolar o efeito de "classificar a imagem inteira" vs "classificar
um recorte pequeno de uma unica regiao", usando exatamente os mesmos
individuos em cada split (nao e uma nova divisao aleatoria).

Uso:
    cd training
    python utils/prepare_wholeimage_data.py
"""
import argparse
import os
import shutil
from collections import defaultdict

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
CLASSES = ["0", "1", "2", "3"]


def collect_split_identities(final_dir):
    """Para cada split/classe, uniao dos nomes de arquivo em todas as
    regioes = conjunto de imagens originais que pertencem a esse split."""
    result = defaultdict(lambda: defaultdict(set))
    for region in REGIONS:
        for split in ["train", "val", "test"]:
            for cls in CLASSES:
                d = os.path.join(final_dir, region, split, cls)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        result[split][cls].add(fn)
    return result


def resolve_source(fn, acne04_dir, manual_dir, cls):
    if fn.startswith("manual_"):
        basename = fn[len("manual_"):]
        src = os.path.join(manual_dir, cls, basename)
        return src if os.path.exists(src) else None
    else:
        src = os.path.join(acne04_dir, fn)
        return src if os.path.exists(src) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="../data/final")
    parser.add_argument("--acne04-dir", default="../../acne_1024")
    parser.add_argument("--manual-dir", default="../data/manual_annotations")
    parser.add_argument("--output", default="../data/wholeimage")
    args = parser.parse_args()

    identities = collect_split_identities(args.final_dir)

    stats = defaultdict(lambda: defaultdict(int))
    missing = 0
    for split, by_cls in identities.items():
        for cls, filenames in by_cls.items():
            dest_dir = os.path.join(args.output, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for fn in filenames:
                src = resolve_source(fn, args.acne04_dir, args.manual_dir, cls)
                if src is None:
                    missing += 1
                    continue
                dest = os.path.join(dest_dir, fn)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
                stats[split][cls] += 1

    print("=" * 50)
    print("DATASET DE IMAGEM INTEIRA (mesma divisao do pipeline de regiao)")
    print("=" * 50)
    total = 0
    for split in ["train", "val", "test"]:
        n_split = sum(stats[split].values())
        total += n_split
        print(f"\n{split.upper()}: {n_split} imagens")
        for cls in CLASSES:
            print(f"  Level {cls}: {stats[split][cls]}")
    print(f"\nTotal: {total} imagens (unicas, sem recorte por regiao)")
    print(f"Imagens de origem nao encontradas: {missing}")


if __name__ == "__main__":
    main()
