
import os
import cv2
import mediapipe as mp
from tqdm import tqdm

COMBINED_DIR = "../../data/combined"
DEST_DIR = "../../data/processed"
OUTPUT_SIZE = (224, 224)

mp_mesh = mp.solutions.face_mesh
mesh = mp_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

REGION_LANDMARKS = {
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323],
    "chin": [152, 377, 400, 378, 379, 365, 397, 288],
    "nose": [1, 2, 98, 327, 168, 195],
    "left_cheek": [50, 101, 234, 93, 132],
    "right_cheek": [280, 425, 411, 307, 320]
}

def save_crop(crop, out_path):
    if crop is not None and crop.size > 0:
        resized = cv2.resize(crop, OUTPUT_SIZE)
        cv2.imwrite(out_path, resized)
        return True
    return False

def crop_region(img, landmarks, indices):
    h, w = img.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), w-1), min(max(ys), h-1)
    if x2 > x1 and y2 > y1:
        return img[y1:y2, x1:x2]
    return None

def fallback_crop(img):
    h, w = img.shape[:2]
    cx1 = int(0.25 * w)
    cy1 = int(0.25 * h)
    cx2 = int(0.75 * w)
    cy2 = int(0.75 * h)
    return img[cy1:cy2, cx1:cx2]

def segment_and_save(image_path, split, label):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Falha ao ler {image_path}')
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mesh.process(img_rgb)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        for region, indices in REGION_LANDMARKS.items():
            crop = crop_region(img, lm, indices)
            out_dir = os.path.join(DEST_DIR, region, split, str(label))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base_name}_{region}.jpg")
            if not save_crop(crop, out_path):
                print(f'Crop invalido em {out_path}')
    else:
        print(f'Face mesh falhou em {image_path}, salvando fallback')
        for region in REGION_LANDMARKS.keys():
            crop = fallback_crop(img)
            out_dir = os.path.join(DEST_DIR, region, split, str(label))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base_name}_{region}_fallback.jpg")
            if not save_crop(crop, out_path):
                print(f'Fallback crop invalido em {out_path}')

def main():
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(COMBINED_DIR, split)
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            for img_name in tqdm(os.listdir(label_dir), desc=f"{split}/{label}"):
                img_path = os.path.join(label_dir, img_name)
                segment_and_save(img_path, split, label)

if __name__ == "__main__":
    main()
