"""
Reconstroi data/final_v2/{regiao}/train/{classe}/ (so "train" como pasta
fixa -- o k-fold reatribui o split de verdade via kfold_splits_v2.json,
a pasta fisica nao importa para isso) usando SO Acne04 (acne_1024/) e
AcneLevel TRATADO (data/acnelevel_augmented/, ver utils/augment_acnelevel.py)
-- descarta manual_annotations por decisao do usuario (2026-10-16).

Prefixa os nomes de arquivo por origem (a04_ / lvl_) para nao colidir (as
duas fontes compartilham ~96% dos mesmos nomes de arquivo, por serem a
mesma foto) -- mas mantem no manifesto a identidade CANONICA (nome base
sem prefixo) para que utils/build_grouped_kfold.py agrupe as duas versoes
da MESMA foto no MESMO fold (senao reintroduz vazamento: a versao Acne04
de uma foto no treino e a versao AcneLevel da MESMA foto no teste).

Uso:
    cd training
    python utils/build_acne04_acnelevel_crops.py --output ../data/final_v2
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


def process_one(img_path, face_mesh, crop_size):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    boxes = region_boxes(landmarks, w, h)
    pil_img = Image.fromarray(img_rgb)
    crops = {}
    for region, (rx0, ry0, rx1, ry1) in boxes.items():
        rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
        rx1, ry1 = min(w, int(rx1)), min(h, int(ry1))
        if (rx1 - rx0) < MIN_CROP_SIZE or (ry1 - ry0) < MIN_CROP_SIZE:
            continue
        crops[region] = pil_img.crop((rx0, ry0, rx1, ry1)).resize((crop_size, crop_size), Image.LANCZOS)
    return crops


def acne04_sources():
    for p in sorted(glob.glob("../acne_1024/levle*_*.jpg")):
        name = os.path.basename(p)
        level = int(name.split("_")[0].replace("levle", ""))
        canonical = name  # ja e o nome canonico
        yield p, level, "a04_" + name, canonical


def acnelevel_sources():
    for level in range(4):
        for p in sorted(glob.glob(f"../data/acnelevel_augmented/{level}/*.jpg")):
            name = os.path.basename(p)
            yield p, level, "lvl_" + name, name  # canonical = mesmo nome (acnelevel preserva o nome original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../data/final_v2")
    ap.add_argument("--crop-size", type=int, default=224)
    args = ap.parse_args()

    manifest = []  # {out_name, canonical, source, level}
    stats = {"a04_ok": 0, "a04_fail": 0, "lvl_ok": 0, "lvl_fail": 0}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as fm:
        for src_iter, src_name in [(acne04_sources(), "a04"), (acnelevel_sources(), "lvl")]:
            for path, level, out_name, canonical in src_iter:
                crops = process_one(path, fm, args.crop_size)
                if not crops:
                    stats[f"{src_name}_fail"] += 1
                    continue
                for region, crop in crops.items():
                    dest_dir = os.path.join(args.output, region, "train", str(level))
                    os.makedirs(dest_dir, exist_ok=True)
                    crop.save(os.path.join(dest_dir, out_name), quality=95)
                manifest.append({"name": out_name, "canonical": canonical, "source": src_name, "level": level})
                stats[f"{src_name}_ok"] += 1
                if sum(stats.values()) % 300 == 0:
                    print(stats, flush=True)

    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("CONCLUIDO:", stats)
    print("total manifest entries:", len(manifest))


if __name__ == "__main__":
    main()
