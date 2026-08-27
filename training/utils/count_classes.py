
import os
from collections import defaultdict

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
regions = ["chin", "forehead", "left_cheek", "nose", "right_cheek"]
splits = ["train", "val", "test"]
classes = ["0", "1", "2", "3"]

def count_images():
    for region in regions:
        print(f'\nRegiao: {region}')
        total_region = 0
        for split in splits:
            print(f"   Conjunto: {split}")
            split_total = 0
            for cls in classes:
                class_dir = os.path.join(DATA_DIR, region, split, cls)
                if not os.path.exists(class_dir):
                    count = 0
                else:
                    count = len([f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
                print(f"     Classe {cls}: {count}")
                split_total += count
            print(f"     Total {split}: {split_total}")
            total_region += split_total
        print(f"Total geral da regiao '{region}': {total_region}")

if __name__ == "__main__":
    count_images()
