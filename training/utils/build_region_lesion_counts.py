"""
Conta, para cada recorte de regiao em data/final_v3, quantas lesoes anotadas
do ACNE04 (Detection/VOC2007/Annotations, 18.983 caixas) caem dentro dele.

Os recortes foram gerados a partir de acne_1024/ (foto original esticada para
1024x1024) com caixas calculadas por landmarks, que nao foram salvas. A posicao
de cada recorte e recuperada por correlacao cruzada normalizada (busca em
largura x altura em 1/4 da resolucao, refinada em 1/2) contra a foto de 1024 px
da mesma identidade fotografica. As caixas de lesao, que estao na resolucao
original, sao levadas para 1024x1024 pela mesma escala independente em x e y.

Uma lesao conta para o recorte quando o centro da caixa cai dentro dele.
Recortes com correlacao abaixo do limiar ficam sem contagem (count = -1) e nao
recebem a loss de contagem no treino.

Uso:
    cd training
    python utils/build_region_lesion_counts.py --workers 16
Saida: results/region_lesion_counts.json
"""
import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
MIN_SCORE = 0.6


def load_boxes(ann_dir):
    boxes = {}
    for f in glob.glob(os.path.join(ann_dir, "Annotations", "*.xml")):
        s = open(f, encoding="utf-8", errors="ignore").read()
        w = int(re.search(r"<width>(\d+)", s).group(1))
        h = int(re.search(r"<height>(\d+)", s).group(1))
        b = re.findall(r"<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*<ymax>(\d+)</ymax>", s)
        centers = [((int(x0) + int(x1)) / 2 * 1024 / w, (int(y0) + int(y1)) / 2 * 1024 / h) for x0, y0, x1, y1 in b]
        boxes[os.path.basename(f)[:-4] + ".jpg"] = centers
    return boxes


def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(g)


def search(photo, crop, scale, sizes, window=None):
    """Retorna (score, x0, y0, w, h) em coordenadas de 1024."""
    ph = cv2.resize(photo, (1024 // scale, 1024 // scale), interpolation=cv2.INTER_AREA)
    best = (-2.0, 0, 0, 0, 0)
    for w, h in sizes:
        tw, th = max(4, round(w / scale)), max(4, round(h / scale))
        if tw >= ph.shape[1] or th >= ph.shape[0]:
            continue
        t = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        if window is None:
            region, ox, oy = ph, 0, 0
        else:
            x0, y0, x1, y1 = [int(v / scale) for v in window]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(ph.shape[1], x1 + tw), min(ph.shape[0], y1 + th)
            region, ox, oy = ph[y0:y1, x0:x1], x0, y0
            if region.shape[0] <= th or region.shape[1] <= tw:
                continue
        r = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, (loc[0] + ox) * scale, (loc[1] + oy) * scale, w, h)
    return best


def locate(args):
    crop_path, photo_path = args
    cv2.setNumThreads(1)
    photo = prep(cv2.resize(cv2.imread(photo_path), (1024, 1024)))
    crop = prep(cv2.imread(crop_path))
    coarse = [(w, h) for w in range(96, 801, 24) for h in range(96, 801, 24)]
    s, x, y, w, h = search(photo, crop, 4, coarse)
    fine = [(w + dw, h + dh) for dw in range(-24, 25, 6) for dh in range(-24, 25, 6) if w + dw >= 96 and h + dh >= 96]
    s2, x2, y2, w2, h2 = search(photo, crop, 2, fine, window=(x - 32, y - 32, x + 32, y + 32))
    if s2 >= s:
        s, x, y, w, h = s2, x2, y2, w2, h2
    return crop_path, float(s), (int(x), int(y), int(w), int(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="../data/final_v3")
    ap.add_argument("--photos", default="../data/acne_1024")
    ap.add_argument("--ann", default="../data/acne04_detection")
    ap.add_argument("--out", default="results/region_lesion_counts.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    centers = load_boxes(args.ann)
    manifest = json.load(open(os.path.join(args.crops, "manifest.json")))
    jobs = []
    for m in manifest:
        photo = os.path.join(args.photos, m["canonical"])
        if m["source"] != "a04" or not os.path.exists(photo):
            continue
        for r in m["regions"]:
            jobs.append((os.path.join(args.crops, r, "train", str(m["level"]), m["name"]), photo))
    print(f"{len(jobs)} recortes para localizar", flush=True)

    out = {}
    with Pool(args.workers) as pool:
        for i, (crop_path, score, (x, y, w, h)) in enumerate(pool.imap_unordered(locate, jobs, chunksize=8)):
            parts = crop_path.replace("\\", "/").split("/")
            region, name = parts[-4], parts[-1]
            canonical = name.split("_", 1)[1]
            c = centers.get(canonical)
            if c is None or score < MIN_SCORE:
                count = -1
            else:
                count = sum(1 for cx, cy in c if x <= cx < x + w and y <= cy < y + h)
            out[f"{region}/{name}"] = {"count": count, "score": round(score, 4), "box": [x, y, w, h]}
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)

    # As copias AcneLevel passaram por uma transformacao geometrica, entao a busca
    # direta na foto original nao e confiavel. Como as regioes sao definidas por
    # landmarks anatomicos, a mesma regiao da mesma foto cobre a mesma pele: o
    # recorte lvl herda a contagem do recorte a04 correspondente, quando existe.
    for m in manifest:
        if m["source"] != "lvl":
            continue
        for r in m["regions"]:
            src = out.get(f"{r}/a04_{m['canonical']}")
            out[f"{r}/{m['name']}"] = {"count": src["count"] if src else -1, "score": None, "box": None,
                                       "from": "a04" if src else None}

    json.dump(out, open(args.out, "w"), indent=1)
    scores = np.array([v["score"] for v in out.values() if v["score"] is not None])
    valid = [v["count"] for v in out.values() if v["count"] >= 0]
    print(f"salvos {len(out)}; com contagem {len(valid)}; score mediano (a04) {np.median(scores):.3f}; "
          f"abaixo do limiar {np.mean(scores < MIN_SCORE):.1%}")
    print(f"contagem por recorte: media {np.mean(valid):.2f}, max {max(valid)}")


if __name__ == "__main__":
    main()
