
import os
import json
import shutil
import random
from tqdm import tqdm

ALL_1024_DIR = "../../all_1024"
DATASET_DIR = "../../Dataset"
DEST_DIR = "../../data/processed"
REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

def parse_all_1024():
    metadata_path = os.path.join(ALL_1024_DIR, "metadata.jsonl")
    files = {0: [], 1: [], 2: [], 3: []}
    with open(metadata_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            fname = data["file_name"]
            label = int(fname.split("_")[0].replace("levle", ""))
            files[label].append(os.path.join(ALL_1024_DIR, fname))
    return files

def parse_dataset_folder():
    files = {0: [], 1: [], 2: [], 3: []}
    for split_folder in ["Train", "Validation"]:
        split_path = os.path.join(DATASET_DIR, split_folder)
        for level in os.listdir(split_path):
            label = int(level.replace("Level", "").strip())
            level_path = os.path.join(split_path, level)
            for img_name in os.listdir(level_path):
                files[label].append(os.path.join(level_path, img_name))
    return files

def distribute_and_copy(all_files):
    for label, paths in all_files.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val = int(n * SPLIT_RATIO["val"])
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:]
        }
        for split, files in splits.items():
            for fpath in tqdm(files, desc=f"{split}/{label}"):
                fname = os.path.basename(fpath)
                for region in REGIONS:
                    dest_dir = os.path.join(DEST_DIR, region, split, str(label))
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, fname)
                    shutil.copy2(fpath, dest_path)

def main():
    files_1024 = parse_all_1024()
    files_dataset = parse_dataset_folder()
    all_files = {k: files_1024.get(k, []) + files_dataset.get(k, []) for k in range(4)}
    distribute_and_copy(all_files)
    print('Distribuicao concluida.')

if __name__ == "__main__":
    main()
