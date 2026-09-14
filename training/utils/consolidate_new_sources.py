"""
Consolida varias fontes de imagens de acne baixadas recentemente
(AcneDataset, data-2, "Skin v2"/acne) num unico lote pronto para upload no
Roboflow -- deduplicando por HASH DE CONTEUDO (nao so nome de arquivo)
contra:
  (a) tudo que ja usamos no projeto (ACNE04 bruto, manual_annotations, SCIN)
  (b) as proprias fontes novas entre si (varios datasets do Kaggle/Roboflow
      reempacotam as mesmas fotos sob nomes diferentes)

Tambem descarta arquivos corrompidos/ilegveis (checagem via PIL) -- filtro
de "possivel de anotar".

Gera um manifesto CSV com a fonte original de cada imagem mantida, para
citar no artigo depois.

Uso:
    cd training
    python utils/consolidate_new_sources.py
"""
import csv
import hashlib
import os
import re
import shutil
from PIL import Image


def base_identity(filename):
    """Remove o sufixo de hash do Roboflow (_jpg.rf.<hash>.jpg etc) para
    identificar copias aumentadas (augmentation) da mesma foto original."""
    return re.sub(r"_(jpg|png|jpeg)\.rf\..*", "", filename, flags=re.I)

BASE = "/Users/GonLu/Documents/Projetos/pibic_final"

EXISTING_DIRS = [
    os.path.join(BASE, "acne_1024"),
    os.path.join(BASE, "data/manual_annotations"),
    os.path.join(BASE, "data/scin_acne_raw"),
]

NEW_SOURCES = [
    ("Acne_Dataset_Image_Kaggle", os.path.join(BASE, "Downloads_ignore")),  # placeholder, overwritten below
]

NEW_SOURCES = [
    ("Acne_Dataset_Image_Kaggle", "/Users/GonLu/Downloads/AcneDataset"),
    ("Acne_Dataset_YOLOv8_Kaggle", "/Users/GonLu/Downloads/data-2"),
    ("Skin_Issues_Dataset_acne_Kaggle", "/Users/GonLu/Downloads/Skin v2/acne"),
]

OUTPUT_DIR = os.path.join(BASE, "data/roboflow_upload_batch")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "_manifest.csv")

IMG_EXT = (".jpg", ".jpeg", ".png")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, fn))
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Indexando imagens ja existentes no projeto (hash)...")
    known_hashes = set()
    for d in EXISTING_DIRS:
        if not os.path.isdir(d):
            print(f"  aviso: {d} nao existe, pulando")
            continue
        files = list_images(d)
        for fp in files:
            try:
                known_hashes.add(md5_of(fp))
            except Exception:
                pass
        print(f"  {d}: {len(files)} imagens indexadas")
    print(f"Total de hashes conhecidos: {len(known_hashes)}")

    manifest_rows = []
    counter = 0
    stats = {}

    for source_name, source_dir in NEW_SOURCES:
        if not os.path.isdir(source_dir):
            print(f"\naviso: fonte '{source_name}' nao encontrada em {source_dir}, pulando")
            continue

        files = list_images(source_dir)
        total = len(files)

        # 1) dedupe por identidade base DENTRO da fonte -- varias destas
        # fontes reaplicam augmentation (2-10x) mesmo em splits val/test,
        # entao isso evita levar copias quase-identicas da mesma foto.
        by_identity = {}
        for fp in files:
            ident = base_identity(os.path.basename(fp))
            by_identity.setdefault(ident, fp)
        intra_dedup_removed = total - len(by_identity)

        acne04_name = 0
        dup_existing = 0
        corrupt = 0
        kept = 0

        for fp in by_identity.values():
            # 2) exclui fotos que ja sao do ACNE04 (mesma convencao de nome
            # "levleN_..."), mesmo que re-exportadas/recomprimidas (hash
            # de conteudo diferente do original, mas mesma foto).
            if re.match(r"levle\d+_", os.path.basename(fp), re.I):
                acne04_name += 1
                continue

            try:
                h = md5_of(fp)
            except Exception:
                corrupt += 1
                continue

            if h in known_hashes:
                dup_existing += 1
                continue

            # valida que a imagem abre corretamente (filtro de qualidade)
            try:
                with Image.open(fp) as im:
                    im.verify()
            except Exception:
                corrupt += 1
                continue

            known_hashes.add(h)
            counter += 1
            ext = os.path.splitext(fp)[1].lower()
            new_name = f"{source_name}_{counter:05d}{ext}"
            dest = os.path.join(OUTPUT_DIR, new_name)
            shutil.copy2(fp, dest)
            manifest_rows.append({
                "new_filename": new_name,
                "source": source_name,
                "original_path": fp,
                "md5": h,
            })
            kept += 1

        stats[source_name] = {
            "total": total, "kept": kept, "dup_existing": dup_existing,
            "corrupt": corrupt, "intra_dedup_removed": intra_dedup_removed,
            "acne04_name": acne04_name,
        }
        print(f"\n{source_name} ({source_dir}):")
        print(f"  total de arquivos: {total}")
        print(f"  copias de augmentation da mesma foto (mesma fonte): -{intra_dedup_removed}")
        print(f"  nome no padrao ACNE04 (levleN_...), excluida: -{acne04_name}")
        print(f"  duplicata de algo que ja temos ou de outra fonte nova (hash): -{dup_existing}")
        print(f"  corrompida/ilegivel: -{corrupt}")
        print(f"  mantida (nova, unica, valida): {kept}")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["new_filename", "source", "original_path", "md5"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50)
    total_kept = sum(s["kept"] for s in stats.values())
    for name, s in stats.items():
        print(f"  {name}: +{s['kept']} novas (de {s['total']} totais)")
    print(f"\nTotal de imagens novas, unicas e validas: {total_kept}")
    print(f"Pasta pronta para upload: {OUTPUT_DIR}")
    print(f"Manifesto (fonte de cada imagem): {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
