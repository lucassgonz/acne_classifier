"""
Quantifica, para cada fonte (Acne04 e AcneLevel), quantas imagens sao
closeup de 1 regiao vs rosto completo -- e, para cada uma das 5 regioes,
quantas imagens tem aquela regiao de fato VISIVEL (caixa calculada dentro
do quadro) vs CORTADA (parte da regiao esperada cai fora da foto, ou seja,
a foto so mostra um pedaco daquela regiao -- o crop atual gera um recorte
enganoso nesses casos, sem indicar isso em lugar nenhum).

Criterio de "visivel com confianca": a caixa da regiao, calculada a partir
dos landmarks (sem recorte pelos limites da imagem), precisa estar quase
toda (>=90% da area) dentro do quadro da foto. Se uma fatia maior que isso
cai fora, a regiao "nao esta realmente ali" -- so uma parte dela.

Uso:
    cd training
    python utils/audit_region_visibility.py
"""
import glob
import json
import os

import cv2
import mediapipe as mp

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
mp_face_mesh = mp.solutions.face_mesh


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def region_boxes_raw(landmarks, w, h):
    """Mesma formula de build_acne04_acnelevel_crops.py, mas SEM clipar
    pelos limites da imagem -- precisamos saber o quanto cada caixa
    extrapola o quadro."""
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


def audit(sources, fm):
    stats = {"no_face": 0, "ok": 0}
    region_visible = {r: 0 for r in REGIONS}
    region_total = {r: 0 for r in REGIONS}
    n_regions_hist = {i: 0 for i in range(6)}
    for path in sources:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            stats["no_face"] += 1
            n_regions_hist[0] += 1
            continue
        stats["ok"] += 1
        landmarks = results.multi_face_landmarks[0].landmark
        boxes = region_boxes_raw(landmarks, w, h)
        n_ok = 0
        for r, box in boxes.items():
            region_total[r] += 1
            if visible_fraction(box, w, h) >= 0.90:
                region_visible[r] += 1
                n_ok += 1
        n_regions_hist[n_ok] += 1
    return stats, region_visible, region_total, n_regions_hist


def main():
    acne04 = sorted(glob.glob("../acne_1024/levle*_*.jpg"))
    acnelevel = []
    for lvl in range(4):
        acnelevel += sorted(glob.glob(f"../Dataset/Train/Level {lvl}/*.jpg"))
        acnelevel += sorted(glob.glob(f"../Dataset/Validation/Level {lvl}/*.jpg"))

    report = {}
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as fm:
        for name, sources in [("acne04", acne04), ("acnelevel", acnelevel)]:
            print(f"=== {name} ({len(sources)} imagens) ===", flush=True)
            stats, region_visible, region_total, hist = audit(sources, fm)
            print("face detectada:", stats)
            print("regioes visiveis (>=90% dentro do quadro) / total com rosto detectado:")
            for r in REGIONS:
                pct = 100 * region_visible[r] / max(1, stats["ok"])
                print(f"  {r}: {region_visible[r]}/{stats['ok']} ({pct:.1f}%)")
            print("histograma de quantas das 5 regioes ficam visiveis por imagem:", hist)
            report[name] = {"stats": stats, "region_visible": region_visible,
                             "total_with_face": stats["ok"], "hist_n_regions": hist}

    with open("results/region_visibility_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Salvo em results/region_visibility_audit.json")


if __name__ == "__main__":
    main()
