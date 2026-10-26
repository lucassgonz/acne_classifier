"""
Pre-processa o AcneLevel (Dataset/) para a ablacao cross-dataset Acne04 x
AcneLevel: Dataset/ e acne_1024/ compartilham ~1406 das mesmas fotos (a
mesma foto, so em resolucao diferente -- confirmado por nome de arquivo e
inspecao visual), entao usa-las como "duas fontes independentes" sem
tratamento faria o teste cross-dataset virar so reconhecimento da mesma
foto. Aqui aplicamos uma transformacao FIXA (determinística por nome de
arquivo, nao aleatoria a cada epoca) de cor, brilho e posicao em cada
imagem do AcneLevel antes de salvar, para que a versao usada na ablacao
nao seja pixel-identica a versao do Acne04.

Limitacao honesta: a pessoa e a lesao fotografada continuam as mesmas --
isso reduz o "match" trivial de pixels, mas nao cria independencia
estatistica de verdade entre as fontes. Resultado cross-dataset deve ser
lido com essa ressalva.

Uso:
    cd training
    python utils/augment_acnelevel.py --input ../Dataset --output ../data/acnelevel_augmented
"""
import argparse
import hashlib
import os
import random

from PIL import Image, ImageEnhance


def seeded_rng(filename):
    h = hashlib.md5(filename.encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def transform_image(img, rng):
    # Cor: hue/saturation via conversão para HSV e deslocamento do canal H.
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    hue_shift = rng.randint(-18, 18)
    h = h.point(lambda p: (p + hue_shift) % 256)
    sat_factor = rng.uniform(0.75, 1.25)
    s = s.point(lambda p: min(255, max(0, int(p * sat_factor))))
    img = Image.merge("HSV", (h, s, v)).convert("RGB")

    # Brilho e contraste.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.80, 1.20))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))

    # Posicao: leve rotacao + crop/zoom deslocado (muda o enquadramento).
    angle = rng.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    w, h_ = img.size
    zoom = rng.uniform(1.05, 1.18)
    new_w, new_h = int(w / zoom), int(h_ / zoom)
    max_dx, max_dy = w - new_w, h_ - new_h
    dx = rng.randint(0, max(0, max_dx))
    dy = rng.randint(0, max(0, max_dy))
    img = img.crop((dx, dy, dx + new_w, dy + new_h)).resize((w, h_), Image.BICUBIC)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../Dataset")
    ap.add_argument("--output", default="../data/acnelevel_augmented")
    args = ap.parse_args()

    n = 0
    for split in ("Train", "Validation"):
        for level in range(4):
            src_dir = os.path.join(args.input, split, f"Level {level}")
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(args.output, str(level))
            os.makedirs(dst_dir, exist_ok=True)
            for fn in os.listdir(src_dir):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                dst_path = os.path.join(dst_dir, fn)
                if os.path.exists(dst_path):
                    n += 1
                    continue
                img = Image.open(os.path.join(src_dir, fn)).convert("RGB")
                rng = seeded_rng(fn)
                img = transform_image(img, rng)
                img.save(dst_path, quality=92)
                n += 1
                if n % 200 == 0:
                    print(f"  {n} processadas...", flush=True)

    print(f"AcneLevel tratado: {n} imagens em {args.output}")


if __name__ == "__main__":
    main()
