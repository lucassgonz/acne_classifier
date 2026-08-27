
import os
from PIL import Image, ImageOps, ImageEnhance
import random
from torchvision import transforms
from tqdm import tqdm

REGIONS = ["chin", "forehead", "left_cheek", "nose", "right_cheek"]
CLASSES_TO_AUGMENT = [2, 3]
SPLITS = ["train"]
AUGS_PER_IMAGE = 5

def augment_image(img):
    transform_ops = [
        lambda x: ImageOps.mirror(x),
        lambda x: x.rotate(random.randint(-20, 20)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: transforms.Resize((224, 224))(transforms.CenterCrop(180)(x))
    ]
    return [random.choice(transform_ops)(img.copy()) for _ in range(AUGS_PER_IMAGE)]

def augment_and_save(image_path, output_dir, base_name, ext):
    img = Image.open(image_path).convert("RGB")
    augmented_images = augment_image(img)
    for i, aug_img in enumerate(augmented_images):
        save_name = f"aug_{i}_{base_name}{ext}"
        save_path = os.path.join(output_dir, save_name)
        aug_img.save(save_path)

def main():
    for region in REGIONS:
        for split in SPLITS:
            for class_id in CLASSES_TO_AUGMENT:
                BASE_DIR = r"C:\Users\GonLu\Documents\Projetos\pibic_final\data\processed"
                class_dir = os.path.join(BASE_DIR, region, split, str(class_id))

                print(f'Verificando {class_dir}...')

                if not os.path.exists(class_dir):
                    print('Pasta nao existe.')
                    continue

                images = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("aug_")]

                if len(images) == 0:
                    print('Nenhuma imagem original encontrada para augmentar.')
                    continue

                print(f'Encontradas {len(images)} imagens originais. Iniciando aumento...')

                for filename in tqdm(images):
                    image_path = os.path.join(class_dir, filename)
                    base_name, ext = os.path.splitext(filename)
                    augment_and_save(image_path, class_dir, base_name, ext)

                total = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f'Total apos aumento: {total} imagens')

if __name__ == "__main__":
    main()
