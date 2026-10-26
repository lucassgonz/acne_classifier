"""
Igual a build_acne04_acnelevel_crops.py (v2), mas com um filtro de
confianca por regiao: SO gera o crop de uma regiao quando a caixa
calculada pelos landmarks (antes de cortar pelas bordas da imagem) esta
>=90% dentro do quadro da foto. Caso contrario, a regiao e descartada
PARA AQUELA IMAGEM (nao gera um crop enganoso cortado na borda).

Motivo (achado por auditoria, utils/audit_region_visibility.py, 2026-10-16):
em ~66% das fotos do Acne04 o queixo nao aparece completo no quadro (so
testa/bochecha), e em ~50% das fotos de ambas as fontes uma das bochechas
fica parcialmente fora -- o pipeline v2 gerava um "crop de queixo"/∂"crop
de bochecha" de qualquer jeito, cortado na borda, essencialmente uma
imagem de fundo/pele generica rotulada como aquela regiao. Esse ruido de
rotulo provavelmente contribuiu pra acuracia baixa da ablacao de
embedding region-crop.

Uso:
    cd training
    python utils/build_acne04_acnelevel_crops_v3.py --output ../data/final_v3
"""
import argparse
import glob
import json
import os

import cv2
import mediapipe as mp
from PIL import Image

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
MIN_CROP_SIZE = 30
MIN_VISIBLE_FRACTION = 0.90
mp_face_mesh = mp.solutions.face_mesh


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def region_boxes(landmarks, w, h):
    top_forehead = get_pt(landmarks, 10, w, h)
    brow_mid = get_pt(landmarks, 9, w, h)
    nose_bridge = get_pt(landmarks, 168, w, h)
    nose_bottom = get_pt(landmarks, 2, w, h)
    chin_bottom = get_pt(landmarks, 152, w, h)
    mouth_bottom = get_pt(landmarks, 17, w, h)
    left_face_edge = get_pt(landmarks, 234, w, h)
    right_face_edge = get_pt(landmarks, 454, w, h)
    left_eye_out = get_pt(landmarks, 33, w, h)
    right_eye_out = get_pt(landmarks, 263, w, h)
    nose_left_wing = get_pt(landmarks, 129, w, h)
    nose_right_wing = get_pt(landmarks, 358, w, h)
    jaw_left = get_pt(landmarks, 172, w, h)
    jaw_right = get_pt(landmarks, 397, w, h)
    face_w = right_face_edge[0] - left_face_edge[0]
    return {
        "forehead": (brow_mid[0] - 0.30 * face_w, top_forehead[1] - 0.05 * face_w,
                     brow_mid[0] + 0.30 * face_w, brow_mid[1] - 0.02 * face_w),
        "nose": (nose_left_wing[0] - 0.03 * face_w, nose_bridge[1],
                 nose_right_wing[0] + 0.03 * face_w, nose_bottom[1] + 0.05 * face_w),
        "chin": (jaw_left[0] + 0.05 * face_w, mouth_bottom[1] + 0.03 * face_w,
                 jaw_right[0] - 0.05 * face_w, chin_bottom[1] + 0.05 * face_w),
        "left_cheek": (left_face_edge[0], left_eye_out[1] + 0.05 * face_w,
                       nose_left_wing[0] - 0.02 * face_w, jaw_left[1]),
        "right_cheek": (nose_right_wing[0] + 0.02 * face_w, right_eye_out[1] + 0.05 * face_w,
                        right_face_edge[0], jaw_right[1]),
    }


def visible_fraction(box, w, h):
    x0, y0, x1, y1 = box
    area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    if area <= 0:
        return 0.0
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    inter = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
    return inter / area


def process_one(img_path, face_mesh, crop_size):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, None
    landmarks = results.multi_face_landmarks[0].landmark
    boxes = region_boxes(landmarks, w, h)
    pil_img = Image.fromarray(img_rgb)
    crops = {}
    skipped_low_conf = []
    for region, box in boxes.items():
        rx0, ry0, rx1, ry1 = box
        if visible_fraction(box, w, h) < MIN_VISIBLE_FRACTION:
            skipped_low_conf.append(region)
            continue
        rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
        rx1, ry1 = min(w, int(rx1)), min(h, int(ry1))
        if (rx1 - rx0) < MIN_CROP_SIZE or (ry1 - ry0) < MIN_CROP_SIZE:
            skipped_low_conf.append(region)
            continue
        crops[region] = pil_img.crop((rx0, ry0, rx1, ry1)).resize((crop_size, crop_size), Image.LANCZOS)
    return crops, skipped_low_conf


def acne04_sources():
    for p in sorted(glob.glob("../acne_1024/levle*_*.jpg")):
        name = os.path.basename(p)
        level = int(name.split("_")[0].replace("levle", ""))
        canonical = name
        yield p, level, "a04_" + name, canonical


def acnelevel_sources():
    for level in range(4):
        for p in sorted(glob.glob(f"../data/acnelevel_augmented/{level}/*.jpg")):
            name = os.path.basename(p)
            yield p, level, "lvl_" + name, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../data/final_v3")
    ap.add_argument("--crop-size", type=int, default=224)
    args = ap.parse_args()

    manifest = []
    stats = {"a04_ok": 0, "a04_fail": 0, "lvl_ok": 0, "lvl_fail": 0}
    skipped_by_region = {r: 0 for r in REGIONS}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as fm:
        for src_iter, src_name in [(acne04_sources(), "a04"), (acnelevel_sources(), "lvl")]:
            for path, level, out_name, canonical in src_iter:
                crops, skipped = process_one(path, fm, args.crop_size)
                if crops is None:
                    stats[f"{src_name}_fail"] += 1
                    continue
                if not crops:
                    stats[f"{src_name}_fail"] += 1
                    continue
                for region, crop in crops.items():
                    dest_dir = os.path.join(args.output, region, "train", str(level))
                    os.makedirs(dest_dir, exist_ok=True)
                    crop.save(os.path.join(dest_dir, out_name), quality=95)
                for r in skipped:
                    skipped_by_region[r] += 1
                manifest.append({"name": out_name, "canonical": canonical, "source": src_name,
                                  "level": level, "regions": sorted(crops.keys())})
                stats[f"{src_name}_ok"] += 1
                if sum(stats.values()) % 300 == 0:
                    print(stats, flush=True)

    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("CONCLUIDO:", stats)
    print("total manifest entries:", len(manifest))
    print("regioes descartadas por baixa confianca (fora do quadro):", skipped_by_region)
    from collections import Counter
    per_region_count = Counter(r for m in manifest for r in m["regions"])
    print("crops validos por regiao:", dict(per_region_count))


if __name__ == "__main__":
    main()
