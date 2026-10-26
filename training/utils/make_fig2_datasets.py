"""
Regenera fig2_.png (amostras do dataset + regioes extraidas) com as
fontes CORRETAS usadas no artigo final: ACNE04 (acne_1024) + AcneLevel
(Dataset/, tratado via augment_acnelevel.py) -- substitui a versao antiga
que ainda mostrava a colecao "Author-Annotated", descartada da
metodologia final.

Uso:
    cd training
    python utils/make_fig2_datasets.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REGIONS = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
REGION_LABELS = ["Forehead", "Left Cheek", "Right Cheek", "Nose", "Chin"]
LEVEL_LABELS = ["0-Clear", "1-Mild", "2-Moderate", "3-Severe"]

THUMB = 140
CROP = 110
PAD = 10
LABEL_H = 22


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def load_fit(path, size):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def build_row(y0, draw_canvas, source_name, whole_images, region_images, canon_level):
    f_small = font(13)
    f_bold = font(15)

    x = PAD
    y_text = y0
    draw = ImageDraw.Draw(draw_canvas)
    draw.text((x, y_text), f"{source_name}", font=f_bold, fill="black")
    y_img = y_text + 22

    # 4 whole-image thumbnails (one per severity level)
    for i, (lvl_path, lvl_label) in enumerate(zip(whole_images, LEVEL_LABELS)):
        thumb = load_fit(lvl_path, THUMB)
        draw_canvas.paste(thumb, (x, y_img))
        draw.text((x, y_img + THUMB + 2), lvl_label, font=f_small, fill="black")
        x += THUMB + PAD

    # separator
    sep_x = x + 6
    draw.line([(sep_x, y_img - 4), (sep_x, y_img + THUMB + 20)], fill="gray", width=1)
    x = sep_x + 16

    draw.text((x, y_text), f"Extracted Regions (level {canon_level})", font=f_bold, fill="black")
    for region_label, path in zip(REGION_LABELS, region_images):
        if path is not None:
            crop = load_fit(path, CROP)
        else:
            crop = Image.new("RGB", (CROP, CROP), "#dddddd")
        yy = y_img + (THUMB - CROP) // 2
        draw_canvas.paste(crop, (x, yy))
        draw.text((x, yy + CROP + 2), region_label, font=f_small, fill="black")
        x += CROP + PAD

    return y_img + THUMB + 24


def main():
    base = Path("..")
    acne04_root = base / "acne_1024"
    acnelevel_root = base / "data" / "acnelevel_augmented"
    crops_root = base / "data" / "final_v3"

    acne04_whole = [acne04_root / f"levle{lvl}_1.jpg" for lvl in range(4)]
    # fallback: if levleX_1.jpg missing, glob first match
    import glob
    for i, p in enumerate(acne04_whole):
        if not p.exists():
            matches = sorted(glob.glob(str(acne04_root / f"levle{i}_*.jpg")))
            acne04_whole[i] = Path(matches[0])

    acnelevel_whole = [acnelevel_root / str(lvl) / "levle0_112.jpg" for lvl in range(4)]
    for i, p in enumerate(acnelevel_whole):
        if not p.exists():
            matches = sorted(glob.glob(str(acnelevel_root / str(i) / "*.jpg")))
            acnelevel_whole[i] = Path(matches[0])

    a04_name, a04_level = "a04_levle0_314.jpg", 0
    lvl_name, lvl_level = "lvl_levle2_113.jpg", 2

    a04_regions = [crops_root / r / "train" / str(a04_level) / a04_name for r in REGIONS]
    a04_regions = [p if p.exists() else None for p in a04_regions]
    lvl_regions = [crops_root / r / "train" / str(lvl_level) / lvl_name for r in REGIONS]
    lvl_regions = [p if p.exists() else None for p in lvl_regions]

    width = PAD * 2 + 4 * (THUMB + PAD) + 20 + 5 * (CROP + PAD) + 40
    height = 2 * (22 + THUMB + 24) + PAD * 3

    canvas = Image.new("RGB", (width, height), "white")
    y = PAD
    y = build_row(y, canvas, "ACNE04", acne04_whole, a04_regions, a04_level)
    y = build_row(y + 10, canvas, "AcneLevel (treated)", acnelevel_whole, lvl_regions, lvl_level)

    out = Path("../papers/cbeb/figures/fig2_.png")
    canvas.save(out)
    print("saved", out, canvas.size)


if __name__ == "__main__":
    main()
