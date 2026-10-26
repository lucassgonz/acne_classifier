"""
Detecta imagens duplicadas e quase duplicadas no dataset via pHash (DCT 32x32,
64 bits), comparando tambem a versao espelhada de cada imagem, e verifica
quantos pares caem em splits diferentes (vazamento).

Alem do pHash, agrupa pelo nome original do arquivo sem o sufixo do Roboflow
(_jpg.rf.<hash>), que identifica copias aumentadas da mesma foto.

Uso:
    cd training
    python utils/phash_duplicates.py --images-dir ../data/wholeimage \
        --output results/phash_report.json --montage-dir /tmp/phash_pairs
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_bits(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    return low > np.median(low)


def bits_to_uint64(bits):
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return np.uint64(value)


def hash_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return bits_to_uint64(phash_bits(img)), bits_to_uint64(phash_bits(cv2.flip(img, 1)))


def hamming_to_all(h, others):
    x = np.bitwise_xor(others, h).view(np.uint8).reshape(len(others), 8)
    return POPCOUNT[x].sum(axis=1)


def base_identity(filename):
    name = re.sub(r"_(jpg|png|jpeg)\.rf\..*", "", filename, flags=re.I)
    return os.path.splitext(name)[0]


def source_of(filename):
    return "manual" if filename.startswith("manual_") else "acne04"


def collect(images_dir):
    items = []
    for path in sorted(glob.glob(os.path.join(images_dir, "*", "*", "*"))):
        if not path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = path.split(os.sep)
        items.append({"path": path, "split": parts[-3], "label": parts[-2], "name": parts[-1]})
    return items


def save_montage(pairs, items, out_path, size=224):
    rows = []
    for i, j, d in pairs:
        a = cv2.resize(cv2.imread(items[i]["path"]), (size, size))
        b = cv2.resize(cv2.imread(items[j]["path"]), (size, size))
        row = np.hstack([a, b])
        cv2.putText(row, f"d={d} {items[i]['split']}|{items[j]['split']}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    if rows:
        cv2.imwrite(out_path, np.vstack(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="../data/wholeimage")
    parser.add_argument("--output", default="results/phash_report.json")
    parser.add_argument("--montage-dir", default=None)
    parser.add_argument("--near-threshold", type=int, default=10)
    args = parser.parse_args()

    items = collect(args.images_dir)
    print(f"Imagens: {len(items)}")

    hashes, flips = [], []
    for it in items:
        h, f = hash_image(it["path"])
        hashes.append(h)
        flips.append(f)
    hashes = np.array(hashes, dtype=np.uint64)
    flips = np.array(flips, dtype=np.uint64)

    n = len(items)
    pairs = []
    for i in range(n - 1):
        d_plain = hamming_to_all(hashes[i], hashes[i + 1:])
        d_flip = hamming_to_all(hashes[i], flips[i + 1:])
        d = np.minimum(d_plain, d_flip)
        for k in np.where(d <= args.near_threshold)[0]:
            pairs.append((i, i + 1 + int(k), int(d[k]), bool(d_flip[k] < d_plain[k])))

    thresholds = [0, 2, 4, 6, 8, 10]
    summary = {"n_images": n, "by_threshold": {}}
    for t in thresholds:
        sel = [p for p in pairs if p[2] <= t]
        cross = [p for p in sel if items[p[0]]["split"] != items[p[1]]["split"]]
        cross_src = [p for p in sel if source_of(items[p[0]]["name"]) != source_of(items[p[1]]["name"])]
        by_src = Counter(
            "-".join(sorted([source_of(items[p[0]]["name"]), source_of(items[p[1]]["name"])])) for p in sel
        )
        summary["by_threshold"][t] = {
            "pairs": len(sel),
            "cross_split_pairs": len(cross),
            "cross_source_pairs": len(cross_src),
            "by_source_pair": dict(by_src),
        }
        print(f"d<={t}: pares={len(sel)}  entre splits={len(cross)}  entre fontes={len(cross_src)}  {dict(by_src)}")

    groups = defaultdict(list)
    for idx, it in enumerate(items):
        groups[base_identity(it["name"])].append(idx)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    cross_base = {k: v for k, v in multi.items() if len({items[i]["split"] for i in v}) > 1}
    summary["same_base_name"] = {"groups": len(multi), "groups_across_splits": len(cross_base)}
    print(f"Mesmo nome-base (copias Roboflow): grupos={len(multi)}  entre splits={len(cross_base)}")

    summary["near_pairs"] = [
        {
            "a": items[i]["path"], "b": items[j]["path"], "distance": d, "flipped_match": fl,
            "split_a": items[i]["split"], "split_b": items[j]["split"],
            "label_a": items[i]["label"], "label_b": items[j]["label"],
        }
        for i, j, d, fl in pairs
    ]
    label_conflict = [p for p in summary["near_pairs"] if p["distance"] <= 4 and p["label_a"] != p["label_b"]]
    summary["label_conflicts_d4"] = len(label_conflict)
    print(f"Pares d<=4 com rotulos diferentes: {len(label_conflict)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    if args.montage_dir:
        os.makedirs(args.montage_dir, exist_ok=True)
        cross_pairs = sorted([p for p in pairs if items[p[0]]["split"] != items[p[1]]["split"]], key=lambda p: p[2])
        for lo, hi in [(0, 2), (3, 6), (7, 10)]:
            sel = [(i, j, d) for i, j, d, _ in cross_pairs if lo <= d <= hi][:8]
            save_montage(sel, items, os.path.join(args.montage_dir, f"cross_split_d{lo}-{hi}.jpg"))
        print(f"Montagens em {args.montage_dir}")


if __name__ == "__main__":
    main()
