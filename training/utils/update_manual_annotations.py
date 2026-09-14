"""
Atualiza data/manual_annotations/ com um novo export do projeto Roboflow
"acne classification" (anotacao de gravidade por 2 dermatologistas).

O export do Roboflow multiplica o TREINO por 3x (augmentation), mas nao
val/test. Todos os arquivos de uma mesma foto original compartilham o
mesmo prefixo antes de "_jpg.rf.<hash>" / "_png.rf.<hash>" -- entao
agrupamos por esse prefixo e mantemos so 1 copia por identidade unica
(evita imagens quase-duplicadas geradas por augmentation entrarem no
pool que depois e re-dividido em treino/val/teste por nos mesmos).

So COPIA fotos novas (nao presentes ainda em data/manual_annotations) --
nunca sobrescreve ou remove anotacoes existentes.

Uso:
    cd training
    python utils/update_manual_annotations.py \
        --export-dir "../../../Downloads/acne classification-2" \
        --target-dir ../data/manual_annotations
"""
import argparse
import os
import re
import shutil
from collections import defaultdict

CLASS_MAP = {"level00": "0", "level01": "1", "level02": "2", "level03": "3"}


def base_identity(filename):
    return re.sub(r"_(jpg|png|jpeg)\.rf\..*", "", filename, flags=re.I)


def collect_unique(export_dir):
    """Um arquivo representativo por (identidade, classe), preferindo
    valid/test (garantidamente sem augmentation) sobre train."""
    by_key = {}
    for split, priority in [("valid", 0), ("test", 0), ("train", 1)]:
        for roboflow_cls, cls in CLASS_MAP.items():
            d = os.path.join(export_dir, split, roboflow_cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                ident = base_identity(fn)
                key = (ident, cls)
                if key not in by_key or priority < by_key[key][1]:
                    by_key[key] = (os.path.join(d, fn), priority)
    return {k: v[0] for k, v in by_key.items()}


def existing_identities(target_dir):
    existing = defaultdict(set)
    for cls in ["0", "1", "2", "3"]:
        d = os.path.join(target_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            existing[cls].add(base_identity(fn))
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--target-dir", default="../data/manual_annotations")
    args = parser.parse_args()

    new_pool = collect_unique(args.export_dir)
    existing = existing_identities(args.target_dir)

    print(f"Identidades unicas no novo export: {len(new_pool)}")
    for cls in ["0", "1", "2", "3"]:
        n = sum(1 for (ident, c) in new_pool if c == cls)
        print(f"  classe {cls}: {n}")

    added = defaultdict(int)
    for (ident, cls), src in new_pool.items():
        if ident in existing[cls]:
            continue
        dest_dir = os.path.join(args.target_dir, cls)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dest)
        added[cls] += 1

    print("\n" + "=" * 50)
    print("NOVAS IMAGENS ADICIONADAS")
    print("=" * 50)
    total_added = 0
    for cls in ["0", "1", "2", "3"]:
        print(f"  classe {cls}: +{added[cls]}  (era {len(existing[cls])}, agora {len(existing[cls]) + added[cls]})")
        total_added += added[cls]
    print(f"\nTotal adicionado: {total_added}")


if __name__ == "__main__":
    main()
