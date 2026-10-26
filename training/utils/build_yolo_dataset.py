"""
Gera um dataset no formato YOLO (imagens + rotulos .txt normalizados) a
partir das MESMAS caixas de regiao que utils/segment_faces_mediapipe.py
calcula via MediaPipe Face Mesh -- usadas aqui como "verdade" (ground
truth) para treinar um detector YOLOv8 que reproduza a mesma segmentacao
de 5 regioes (testa/nariz/queixo/bochecha esq/dir).

Objetivo (item do revisor): treinar o YOLOv8 que faltava no repositorio
(pesos originais nao foram salvos), para depois medir (a) o quanto as
caixas do YOLO divergem das do MediaPipe (erro de deteccao) e (b) o
quanto esse erro de deteccao muda a classificacao final de gravidade
quando se usa o recorte do YOLO em vez do recorte do MediaPipe.

Usa os mesmos splits do kfold_splits.json (por imagem) para nao misturar
treino/val/test do classificador com o do detector.

Uso:
    cd training
    python utils/build_yolo_dataset.py --output ../data/yolo_regions
"""
import argparse
import glob
import json
import os

import cv2
import mediapipe as mp
from PIL import Image

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
CLASS_IDX = {r: i for i, r in enumerate(REGIONS)}

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


def whole_image_index():
    index = {}
    for split in ("Train", "Validation"):
        for level in range(4):
            for p in glob.glob(f"../Dataset/{split}/Level {level}/*.jpg"):
                index[os.path.basename(p)] = p
    for p in glob.glob("../data/manual_annotations/*/*.jpg"):
        name = "manual_" + os.path.basename(p)
        index[name] = p
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../data/yolo_regions")
    ap.add_argument("--splits", default="results/kfold_splits.json")
    args = ap.parse_args()

    with open(args.splits) as f:
        folds = json.load(f)["folds"][0]  # fold 0 define um unico split fixo p/ o detector
    index = whole_image_index()

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "no_face": 0, "missing": 0}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=False, min_detection_confidence=0.5) as fm:
        for split in ("train", "val", "test"):
            for name in folds[split]:
                if name not in index:
                    stats["missing"] += 1
                    continue
                img_path = index[name]
                img = cv2.imread(img_path)
                if img is None:
                    stats["missing"] += 1
                    continue
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = fm.process(img_rgb)
                if not results.multi_face_landmarks:
                    stats["no_face"] += 1
                    continue
                landmarks = results.multi_face_landmarks[0].landmark
                boxes = region_boxes(landmarks, w, h)

                stem = os.path.splitext(name)[0]
                dst_img = os.path.join(args.output, "images", split, f"{stem}.jpg")
                Image.open(img_path).convert("RGB").save(dst_img, quality=90)

                lines = []
                for region, (x0, y0, x1, y1) in boxes.items():
                    x0, x1 = max(0.0, min(x0, x1)), min(w, max(x0, x1))
                    y0, y1 = max(0.0, min(y0, y1)), min(h, max(y0, y1))
                    if x1 - x0 < 5 or y1 - y0 < 5:
                        continue
                    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                    bw, bh = (x1 - x0) / w, (y1 - y0) / h
                    lines.append(f"{CLASS_IDX[region]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                dst_label = os.path.join(args.output, "labels", split, f"{stem}.txt")
                with open(dst_label, "w") as lf:
                    lf.write("\n".join(lines))
                stats[split] += 1

    with open(os.path.join(args.output, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.output)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(REGIONS)}\n")
        f.write(f"names: {REGIONS}\n")

    print("Dataset YOLO gerado:", stats)


if __name__ == "__main__":
    main()
