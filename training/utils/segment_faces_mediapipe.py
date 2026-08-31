"""
Segmenta imagens de rosto inteiro em 5 regioes faciais (testa, nariz,
queixo, bochecha esquerda, bochecha direita) usando MediaPipe Face Mesh.

O detector YOLOv8 customizado original (usado para o ACNE04) nao teve os
pesos treinados salvos no repositorio -- essa e uma alternativa baseada
em landmarks faciais (478 pontos do MediaPipe), amplamente usada e bem
estabelecida, para extrair as mesmas 5 regioes a partir da caixa do
rosto detectado (proporcoes geometricas relativas, robustas a variacao
de pose/enquadramento nas fotos de selfie).

Uso:
    cd training
    python utils/segment_faces_mediapipe.py \
        --input-dir ../data/manual_annotations \
        --output-dir ../data/manual_crops \
        --crop-size 224
"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
MIN_CROP_SIZE = 30
MIN_SHARPNESS = 6.0  # variancia do Laplaciano; calibrado empiricamente (crops ruins ~5.8, bons >=10)


def is_sharp_enough(pil_crop):
    gray = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= MIN_SHARPNESS

mp_face_mesh = mp.solutions.face_mesh


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def region_boxes(landmarks, w, h):
    """
    Deriva 5 caixas de regiao a partir de pontos de referencia especificos
    do MediaPipe Face Mesh (nao apenas a caixa geral do rosto) -- calibrado
    visualmente para lidar com selfies em angulos variados. Ver
    training/utils/README de calibracao para o processo de validacao.
    """
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
        "forehead": (
            brow_mid[0] - 0.30 * face_w, top_forehead[1] - 0.05 * face_w,
            brow_mid[0] + 0.30 * face_w, brow_mid[1] - 0.02 * face_w,
        ),
        "nose": (
            nose_left_wing[0] - 0.03 * face_w, nose_bridge[1],
            nose_right_wing[0] + 0.03 * face_w, nose_bottom[1] + 0.05 * face_w,
        ),
        "chin": (
            jaw_left[0] + 0.05 * face_w, mouth_bottom[1] + 0.03 * face_w,
            jaw_right[0] - 0.05 * face_w, chin_bottom[1] + 0.05 * face_w,
        ),
        "left_cheek": (
            left_face_edge[0], left_eye_out[1] + 0.05 * face_w,
            nose_left_wing[0] - 0.02 * face_w, jaw_left[1],
        ),
        "right_cheek": (
            nose_right_wing[0] + 0.02 * face_w, right_eye_out[1] + 0.05 * face_w,
            right_face_edge[0], jaw_right[1],
        ),
    }


def process_image(img_path, face_mesh, crop_size):
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
        # Descarta caixas degeneradas (poucos pixels) -- ficam borradas ao
        # esticar para crop_size. Mesmo limiar do pipeline YOLOv8 original.
        if (rx1 - rx0) < MIN_CROP_SIZE or (ry1 - ry0) < MIN_CROP_SIZE:
            continue
        crop = pil_img.crop((rx0, ry0, rx1, ry1)).resize((crop_size, crop_size), Image.LANCZOS)
        if not is_sharp_enough(crop):
            continue
        crops[region] = crop
    return crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/manual_annotations")
    parser.add_argument("--output-dir", default="../data/manual_crops")
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()

    stats = {r: 0 for r in REGIONS}
    failed = 0
    total = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        for cls in ["0", "1", "2", "3"]:
            cls_dir = os.path.join(args.input_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f"Classe {cls}: {len(files)} imagens")

            for fn in files:
                total += 1
                img_path = os.path.join(cls_dir, fn)
                crops = process_image(img_path, face_mesh, args.crop_size)
                if crops is None:
                    failed += 1
                    continue
                base_name = os.path.splitext(fn)[0]
                for region, crop in crops.items():
                    dest_dir = os.path.join(args.output_dir, region, cls)
                    os.makedirs(dest_dir, exist_ok=True)
                    crop.save(os.path.join(dest_dir, f"{base_name}.jpg"), quality=95)
                    stats[region] += 1

    print("\n" + "=" * 50)
    print("SEGMENTACAO CONCLUIDA")
    print("=" * 50)
    print(f"Total de imagens processadas: {total}")
    print(f"Falhas (rosto nao detectado): {failed}")
    print("Crops gerados por regiao:")
    for r in REGIONS:
        print(f"  {r}: {stats[r]}")
    print(f"Total de crops: {sum(stats.values())}")


if __name__ == "__main__":
    main()
