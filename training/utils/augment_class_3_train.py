
import os
from PIL import Image, ImageOps, ImageEnhance
import random
from torchvision import transforms
from tqdm import tqdm

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
TARGET_CLASS = "3"
SPLIT = "train"
AUGS_PER_IMAGE = 10  
BASE_PATH = r"../../data/processed"

def augment_image(img):
    ops = [
        lambda x: ImageOps.mirror(x),
        lambda x: x.rotate(random.randint(-25, 25)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.75, 1.25)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.75, 1.25)),
        lambda x: transforms.Resize((224, 224))(transforms.CenterCrop(160)(x))
    ]
    return [random.choice(ops)(img.copy()) for _ in range(AUGS_PER_IMAGE)]

def augment_and_save(image_path, output_dir, base_name, ext):
    img = Image.open(image_path).convert("RGB")
    augmented = augment_image(img)
    for i, aug in enumerate(augmented):
        save_name = f"aug3_{i}_{base_name}{ext}"
        save_path = os.path.join(output_dir, save_name)
        aug.save(save_path)

def main():
    for region in REGIONS:
        class_dir = os.path.join(BASE_PATH, region, SPLIT, TARGET_CLASS)
        if not os.path.exists(class_dir):
            print(f'Pasta nao encontrada: {class_dir}')
            continue

        print(f'Aumentando classe 3 na regiao {region}...')
        for filename in tqdm(os.listdir(class_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if filename.startswith("aug3_"):
                continue
            path = os.path.join(class_dir, filename)
            base_name, ext = os.path.splitext(filename)
            augment_and_save(path, class_dir, base_name, ext)

        total = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f'Total de imagens apos aumento em {region}: {total}')

if __name__ == "__main__":
    main()
