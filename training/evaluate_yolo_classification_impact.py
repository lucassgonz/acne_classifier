"""
Item do revisor: quantifica o quanto o erro de deteccao do YOLOv8 (treinado
em utils/build_yolo_dataset.py / run_yolo_train.py para reproduzir as
caixas do MediaPipe) afeta a classificacao final de gravidade, comparando:

  (a) recorte via MediaPipe (referencia "confiavel", usada no pipeline/app)
  (b) recorte via YOLOv8 treinado

...passados pelo MESMO classificador ja treinado (ensemble ONNX de 5
seeds, mobile_models/), nas mesmas imagens de teste do detector. A
diferenca de acuracia entre (a) e (b) e a "contribuicao do erro do YOLO"
pedida pelo revisor.

Uso:
    cd training
    python evaluate_yolo_classification_impact.py
"""
import glob
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
mp_face_mesh = mp.solutions.face_mesh


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def mediapipe_boxes(landmarks, w, h):
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


def crop_to_tensor(pil_img, box, w, h):
    x0, y0, x1, y1 = box
    x0, x1 = max(0, min(int(x0), int(x1))), min(w, max(int(x0), int(x1)))
    y0, y1 = max(0, min(int(y0), int(y1))), min(h, max(int(y0), int(y1)))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = pil_img.crop((x0, y0, x1, y1)).resize((224, 224), Image.BICUBIC)
    arr = np.asarray(crop).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def load_label(name):
    # Nome do teste YOLO ja remete ao mesmo nome usado no kfold_splits.json;
    # label = classe do primeiro recorte correspondente em data/final.
    import glob as g
    for region in REGIONS:
        for cls in range(4):
            if g.glob(f"../data/final/{region}/*/{cls}/{name}"):
                return cls
    return None


def main():
    sessions = [
        ort.InferenceSession(p) for p in sorted(glob.glob("../mobile_models/region_model_seed*.onnx"))
    ]
    print(f"{len(sessions)} modelos ONNX carregados")

    yolo = YOLO("results/yolo/region_detector/weights/best.pt")

    test_images = sorted(glob.glob("../data/yolo_regions/images/test/*.jpg"))
    print(f"{len(test_images)} imagens de teste")

    mp_correct = 0
    yolo_correct = 0
    mp_total = 0
    yolo_total = 0
    mp_region_count = []
    yolo_region_count = []
    disagreements = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        for img_path in test_images:
            name = os.path.basename(img_path)
            label = load_label(name)
            if label is None:
                continue

            img_bgr = cv2.imread(img_path)
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # --- MediaPipe (referencia) ---
            results = fm.process(img_rgb)
            mp_preds = []
            if results.multi_face_landmarks:
                boxes = mediapipe_boxes(results.multi_face_landmarks[0].landmark, w, h)
                for region, box in boxes.items():
                    tensor = crop_to_tensor(pil_img, box, w, h)
                    if tensor is None:
                        continue
                    region_idx = REGIONS.index(region)
                    probs = np.mean([
                        s.run(None, {"image": tensor[None], "region_idx": np.array([region_idx], dtype=np.int64)})[0][0]
                        for s in sessions
                    ], axis=0)
                    mp_preds.append(probs)
            if mp_preds:
                overall = int(np.argmax(np.mean(mp_preds, axis=0)))
                mp_total += 1
                mp_correct += int(overall == label)
                mp_region_count.append(len(mp_preds))

            # --- YOLO ---
            yres = yolo.predict(img_path, verbose=False, device="cpu")[0]
            yolo_preds = []
            seen_classes = set()
            for box, cls_idx in zip(yres.boxes.xyxy.tolist(), yres.boxes.cls.tolist()):
                cls_idx = int(cls_idx)
                if cls_idx in seen_classes:
                    continue
                seen_classes.add(cls_idx)
                tensor = crop_to_tensor(pil_img, tuple(box), w, h)
                if tensor is None:
                    continue
                probs = np.mean([
                    s.run(None, {"image": tensor[None], "region_idx": np.array([cls_idx], dtype=np.int64)})[0][0]
                    for s in sessions
                ], axis=0)
                yolo_preds.append(probs)
            if yolo_preds:
                overall = int(np.argmax(np.mean(yolo_preds, axis=0)))
                yolo_total += 1
                yolo_correct += int(overall == label)
                yolo_region_count.append(len(yolo_preds))

            if mp_preds and yolo_preds:
                mp_overall = int(np.argmax(np.mean(mp_preds, axis=0)))
                yolo_overall = int(np.argmax(np.mean(yolo_preds, axis=0)))
                if mp_overall != yolo_overall:
                    disagreements.append(name)

    report = {
        "n_test_images": len(test_images),
        "mediapipe": {
            "n_evaluated": mp_total,
            "accuracy": mp_correct / mp_total if mp_total else None,
            "avg_regions_detected": float(np.mean(mp_region_count)) if mp_region_count else None,
        },
        "yolo": {
            "n_evaluated": yolo_total,
            "accuracy": yolo_correct / yolo_total if yolo_total else None,
            "avg_regions_detected": float(np.mean(yolo_region_count)) if yolo_region_count else None,
        },
        "disagreement_rate": len(disagreements) / min(mp_total, yolo_total) if min(mp_total, yolo_total) else None,
        "yolo_detection_metrics_from_training": {
            "precision": 0.8571423400096789,
            "recall": 0.8767579164663871,
            "mAP50": 0.9003825547900055,
            "mAP50_95": 0.6481046683891861,
        },
    }
    with open("results/yolo_classification_impact.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
