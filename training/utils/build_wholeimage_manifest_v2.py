"""
Gera manifest.json para o pipeline de IMAGEM INTEIRA usando SO Acne04
(../acne_1024/) e AcneLevel TRATADO (../data/acnelevel_augmented/, ver
augment_acnelevel.py) -- descarta manual_annotations e a versao Dataset/
crua (nao tratada), por decisao do usuario (2026-10-16): usar so Acne04 +
AcneLevel(tratado) em TODOS os testes daqui pra frente.

Nao copia nem reprocessa imagens -- so indexa os caminhos existentes, pois
aqui nao ha recorte de regiao (a tarefa eh classificar a foto inteira).

Prefixa por origem (a04_/lvl_) para evitar colisao de nome de arquivo (as
duas fontes compartilham ~96% dos mesmos nomes), mas guarda a identidade
CANONICA (nome base sem prefixo) para build_kfold_splits_wholeimage_v2.py
agrupar as duas versoes da MESMA foto no MESMO fold.

Uso:
    cd training
    python utils/build_wholeimage_manifest_v2.py
"""
import argparse
import glob
import json
import os


def acne04_sources(root):
    for p in sorted(glob.glob(os.path.join(root, "levle*_*.jpg"))):
        name = os.path.basename(p)
        level = int(name.split("_")[0].replace("levle", ""))
        yield p, level, "a04_" + name, name


def acnelevel_sources(root):
    for level in range(4):
        for p in sorted(glob.glob(os.path.join(root, str(level), "*.jpg"))):
            name = os.path.basename(p)
            yield p, level, "lvl_" + name, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acne04-root", default="../acne_1024")
    ap.add_argument("--acnelevel-root", default="../data/acnelevel_augmented")
    ap.add_argument("--output", default="../data/final_wholeimage_v2/manifest.json")
    args = ap.parse_args()

    manifest = []
    for path, level, name, canonical in acne04_sources(args.acne04_root):
        manifest.append({"name": name, "path": path, "canonical": canonical, "source": "a04", "level": level})
    for path, level, name, canonical in acnelevel_sources(args.acnelevel_root):
        manifest.append({"name": name, "path": path, "canonical": canonical, "source": "lvl", "level": level})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    from collections import Counter
    print("total:", len(manifest), "| por fonte:", Counter(m["source"] for m in manifest),
          "| por nivel:", dict(sorted(Counter(m["level"] for m in manifest).items())))
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
