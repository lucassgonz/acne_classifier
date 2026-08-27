
import os
import json
import shutil
import random
from tqdm import tqdm

ALL_1024_DIR = "../../all_1024"
DATASET_DIR = "../../Dataset"
DEST_DIR = "../../data/combined"
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

def parse_all_1024():
    metadata_path = os.path.join(ALL_1024_DIR, "metadata.jsonl")
    files = []
    with open(metadata_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            fname = data["file_name"]
            label = int(fname.split("_")[0].replace("levle", ""))
            files.append((os.path.join(ALL_1024_DIR, fname), label))
    return files

def parse_dataset_folder():
    files = []
    for split_folder in ["Train", "Validation"]:
        split_path = os.path.join(DATASET_DIR, split_folder)
        for level in os.listdir(split_path):
            label = int(level.replace("Level", "").strip())
            level_path = os.path.join(split_path, level)
            for img_name in os.listdir(level_path):
                files.append((os.path.join(level_path, img_name), label))
    return files

def distribute_and_copy(files):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split, items in splits.items():
        for src_path, label in tqdm(items, desc=f"{split}"):
            dest_dir = os.path.join(DEST_DIR, split, str(label))
            os.makedirs(dest_dir, exist_ok=True)
            fname = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, fname)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)

def main():
    files_1024 = parse_all_1024()
    files_dataset = parse_dataset_folder()
    combined = files_1024 + files_dataset

    unique = {}
    for path, label in combined:
        key = os.path.basename(path).lower()
        if key not in unique:
            unique[key] = (path, label)

    files_unique = list(unique.values())
    print(f'Total de arquivos unicos: {len(files_unique)}')

    distribute_and_copy(files_unique)
    print('Distribuicao concluida.')

if __name__ == "__main__":
    main()
