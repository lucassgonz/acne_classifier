"""
Experimento cross-dataset: treina em um dataset, testa no outro.

Os datasets ACNE04 (acne_1024/) e Acne Level (Dataset/) contêm fotos
distintas dos mesmos pacientes. O ID do paciente é o número no nome do
arquivo (ex: levle1_492.jpg → paciente 492). O split é feito por ID de
paciente para garantir que o mesmo indivíduo não apareça em train e test.

Configurações:
  A → treina no ACNE04, testa no Acne Level
  B → treina no Acne Level, testa no ACNE04

Uso:
    cd training
    python run_cross_dataset.py \
        --acne04-dir ../../acne_1024 \
        --acnelevel-dir ../../Dataset \
        --yolo-crops-dir ../../data/crops \
        --direction A          # ou B, ou both
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
CLASS_NAMES = ["Level_0", "Level_1", "Level_2", "Level_3"]


class ResNet18WithEmbed(nn.Module):
    def __init__(self, num_classes=4, num_regions=5):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.region_embed = nn.Embedding(num_regions, 64)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, region_idx):
        feat = self.backbone(x).flatten(1)
        emb = self.region_embed(region_idx.long()).squeeze(1)
        return self.classifier(torch.cat([feat, emb], dim=1))


# ---------------------------------------------------------------------------
# Utilitários de dataset
# ---------------------------------------------------------------------------

def patient_id_from_filename(filename: str) -> str:
    """Extrai o ID do paciente do nome do arquivo (ex: levle1_492.jpg → '492')."""
    m = re.search(r"levle\d+_(\d+)", filename, re.IGNORECASE)
    return m.group(1) if m else filename


def label_from_filename(filename: str) -> int:
    """Extrai a classe de gravidade do nome do arquivo (ex: levle1_492.jpg → 1)."""
    m = re.match(r"levle(\d+)_", filename, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def collect_acne04_images(acne04_dir: str) -> list:
    """
    Retorna lista de (caminho, label, patient_id) para todas as imagens do ACNE04.
    """
    samples = []
    meta_path = os.path.join(acne04_dir, "metadata.jsonl")
    with open(meta_path) as f:
        for line in f:
            fn = json.loads(line.strip())["file_name"]
            label = label_from_filename(fn)
            pid = patient_id_from_filename(fn)
            if label >= 0:
                samples.append((os.path.join(acne04_dir, fn), label, pid))
    return samples


def collect_acnelevel_images(acnelevel_dir: str) -> list:
    """
    Retorna lista de (caminho, label, patient_id) para todas as imagens do Acne Level.
    """
    samples = []
    for split in ["Train", "Validation"]:
        split_path = os.path.join(acnelevel_dir, split)
        if not os.path.isdir(split_path):
            continue
        for level_dir in os.listdir(split_path):
            level_path = os.path.join(split_path, level_dir)
            if not os.path.isdir(level_path):
                continue
            label = int(level_dir.replace("Level", "").strip())
            for fn in os.listdir(level_path):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    pid = patient_id_from_filename(fn)
                    samples.append((os.path.join(level_path, fn), label, pid))
    return samples


def split_by_patient(samples: list, val_ratio=0.15, seed=42) -> tuple:
    """
    Divide em train/val por ID de paciente (sem data leakage por indivíduo).
    Retorna (train_samples, val_samples).
    """
    rng = random.Random(seed)
    patient_ids = sorted(set(s[2] for s in samples))
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_ids = set(patient_ids[:n_val])
    train = [s for s in samples if s[2] not in val_ids]
    val = [s for s in samples if s[2] in val_ids]
    return train, val


# ---------------------------------------------------------------------------
# Dataset com crops do YOLOv8
# ---------------------------------------------------------------------------

def find_crop(crops_dir: str, patient_id: str, region: str, label: int):
    """
    Procura o crop gerado pelo YOLOv8 para um paciente/região específicos.
    Estrutura esperada: crops_dir/{region}/{label}/levleX_{patient_id}.jpg
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        for cls in range(4):
            path = os.path.join(crops_dir, region, str(cls), f"levle{label}_{patient_id}{ext}")
            if os.path.exists(path):
                return path
    return None


class CrossDatasetDataset(Dataset):
    """
    Dataset para o experimento cross-dataset.
    Usa os crops do YOLOv8 (em crops_dir) se disponíveis;
    caso contrário, usa a imagem facial completa (fallback).
    """

    def __init__(self, samples: list, crops_dir: str, transform=None):
        self.transform = transform
        self.items = []

        for img_path, label, pid in samples:
            fn = os.path.splitext(os.path.basename(img_path))[0]
            for r_idx, region in enumerate(REGIONS):
                crop_path = find_crop(crops_dir, pid, region, label)
                src_path = crop_path if crop_path else img_path
                self.items.append((src_path, label, r_idx))

        print(f"  {len(self.items)} amostras (crops/imagens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, region_idx = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, torch.tensor(region_idx)


# ---------------------------------------------------------------------------
# Treino / Avaliação
# ---------------------------------------------------------------------------

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, labels, regions = imgs.to(device), labels.to(device), regions.to(device)
        preds = model(imgs, regions).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES[:len(set(all_labels))],
        zero_division=0,
    )
    return {"accuracy": acc, "f1_weighted": f1, "report": report}


def train_and_evaluate(
    train_samples, val_samples, test_samples,
    crops_dir, device, batch_size=16, num_epochs=40, patience=10, seed=42,
):
    set_seed(seed)
    train_tf, val_tf = build_transforms()

    train_ds = CrossDatasetDataset(train_samples, crops_dir, train_tf)
    val_ds = CrossDatasetDataset(val_samples, crops_dir, val_tf)
    test_ds = CrossDatasetDataset(test_samples, crops_dir, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    all_labels_train = [s[1] for s in train_ds.items]
    cw = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(4), y=all_labels_train),
        dtype=torch.float,
    ).to(device)

    model = ResNet18WithEmbed().to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for imgs, labels, regions in train_loader:
            imgs, labels, regions = imgs.to(device), labels.to(device), regions.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, regions), labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping na época {epoch}")
                break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acne04-dir", default="../../acne_1024")
    parser.add_argument("--acnelevel-dir", default="../../Dataset")
    parser.add_argument("--yolo-crops-dir", default="../../data/crops",
                        help="Diretório com crops do YOLOv8 (data/crops/{region}/{class}/)")
    parser.add_argument("--direction", choices=["A", "B", "both"], default="both",
                        help="A=treina ACNE04/testa AcneLevel, B=inverso")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/cross_dataset_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    print("Coletando imagens ACNE04...")
    acne04 = collect_acne04_images(args.acne04_dir)
    print(f"  {len(acne04)} imagens, {len(set(s[2] for s in acne04))} pacientes")

    print("Coletando imagens Acne Level...")
    acnelevel = collect_acnelevel_images(args.acnelevel_dir)
    print(f"  {len(acnelevel)} imagens, {len(set(s[2] for s in acnelevel))} pacientes")

    # Pacientes em comum (para o split ser consistente entre datasets)
    common_ids = set(s[2] for s in acne04) & set(s[2] for s in acnelevel)
    print(f"\nPacientes em comum nos dois datasets: {len(common_ids)}")

    results = {}
    crops_dir = args.yolo_crops_dir

    directions = []
    if args.direction in ("A", "both"):
        directions.append(("A", "ACNE04 → Acne Level", acne04, acnelevel))
    if args.direction in ("B", "both"):
        directions.append(("B", "Acne Level → ACNE04", acnelevel, acne04))

    for key, desc, train_src, test_src in directions:
        print(f"\n{'='*60}")
        print(f"Direção {key}: {desc}")
        print(f"{'='*60}")

        train_samples, val_samples = split_by_patient(train_src, val_ratio=0.15, seed=args.seed)
        # Teste: todas as imagens do outro dataset
        # Exclui pacientes presentes no train para evitar leakage por indivíduo
        train_ids = set(s[2] for s in train_samples)
        test_samples = [s for s in test_src if s[2] not in train_ids]

        print(f"Train: {len(train_samples)} imgs | Val: {len(val_samples)} imgs | Test: {len(test_samples)} imgs")
        print(f"Pacientes train: {len(train_ids)} | Pacientes test: {len(set(s[2] for s in test_samples))}")

        metrics = train_and_evaluate(
            train_samples, val_samples, test_samples,
            crops_dir=crops_dir,
            device=device,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            seed=args.seed,
        )

        results[key] = {
            "description": desc,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
        }

        print(f"\nResultado {key}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1-score : {metrics['f1_weighted']:.4f}")
        print("\nRelatório por classe:")
        print(metrics["report"])

    print("\n" + "="*60)
    print("RESUMO CROSS-DATASET")
    print("="*60)
    for key, r in results.items():
        print(f"{r['description']}: Acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {args.output}")


if __name__ == "__main__":
    main()
