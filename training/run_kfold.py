"""
Validacao cruzada em K folds (por imagem original, sem vazamento entre
regioes/splits) usando o melhor setup encontrado (SCIN v2 + Mixup +
embedding de regiao). Diferente do split fixo 70/15/15 usado nos demais
scripts, aqui TODOS os crops sao redistribuidos em K folds e cada fold
serve de teste uma vez -- o que da uma estimativa mais robusta (usa 100%
dos dados para teste, cumulativamente) e produz K modelos naturalmente
diversos (cada um viu uma fatia diferente dos dados) para ensemble.

Uso:
    cd training
    python run_kfold.py --data-dir ../data/final --k 5 \
        --pretrained-backbone models/backbone_scin_pretrained_v2.pth \
        --mixup-alpha 0.2 --save-models-dir models/kfold_ckpt
"""
import argparse
import gc
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from run_ablation import (
    REGIONS, CLASS_NAMES, ResNet18WithEmbed, FocalLoss,
    build_transforms, train_one_epoch, evaluate as eval_batch,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Coleta de todos os crops (ignorando o split fixo train/val/test atual)
# e mapeamento para a imagem original, para o fold ser definido por imagem
# ---------------------------------------------------------------------------

def collect_all_crops(data_dir):
    """Retorna lista de dicts {path, label, region_idx, image_id}."""
    items = []
    for region in REGIONS:
        region_dir = os.path.join(data_dir, region)
        if not os.path.isdir(region_dir):
            continue
        for split in os.listdir(region_dir):
            split_dir = os.path.join(region_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for cls in os.listdir(split_dir):
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fn in os.listdir(cls_dir):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    m = re.match(r"(levle\d+_\d+)", fn, re.I)
                    image_id = m.group(1) if m else fn
                    items.append({
                        "path": os.path.join(cls_dir, fn),
                        "label": int(cls),
                        "region_idx": REGIONS.index(region),
                        "image_id": image_id,
                    })
    return items


def assign_folds(items, k, seed=42):
    """Atribui cada IMAGEM (nao crop) a um fold, estratificado por classe."""
    by_image = {}
    for it in items:
        by_image.setdefault(it["image_id"], it["label"])  # label da imagem (crops da mesma imagem tem mesma classe)

    by_class = {}
    for image_id, label in by_image.items():
        by_class.setdefault(label, []).append(image_id)

    rng = random.Random(seed)
    fold_of_image = {}
    for label, image_ids in by_class.items():
        rng.shuffle(image_ids)
        for i, image_id in enumerate(image_ids):
            fold_of_image[image_id] = i % k

    for it in items:
        it["fold"] = fold_of_image[it["image_id"]]
    return items


class CropListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"]).convert("RGB")
        img = self.transform(img)
        return img, it["label"], torch.tensor(it["region_idx"])


# ---------------------------------------------------------------------------
# Treino de um fold
# ---------------------------------------------------------------------------

def train_fold(train_items, val_items, test_items, device, args):
    train_tf, val_tf = build_transforms(strong_aug=False)
    train_ds = CropListDataset(train_items, train_tf)
    val_ds = CropListDataset(val_items, val_tf)
    test_ds = CropListDataset(test_items, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_labels = [it["label"] for it in train_items]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels),
        dtype=torch.float,
    ).to(device)

    model = ResNet18WithEmbed(pretrained_backbone_path=args.pretrained_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=args.mixup_alpha)
        val_acc, val_f1, val_qwk = eval_batch(model, val_loader, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    test_acc, test_f1, test_qwk = eval_batch(model, test_loader, device)

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return test_acc, test_f1, test_qwk, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Roda so este fold (para isolar em processo separado)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--save-models-dir", default=None)
    parser.add_argument("--output", default="results/kfold_results.json")
    parser.add_argument("--val-fraction-of-train", type=float, default=0.15)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)

    items = collect_all_crops(args.data_dir)
    print(f"Total de crops: {len(items)}", flush=True)
    items = assign_folds(items, args.k, seed=42)

    n_images = len(set(it["image_id"] for it in items))
    print(f"Total de imagens unicas: {n_images}", flush=True)

    folds_to_run = [args.fold] if args.fold is not None else list(range(args.k))

    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = []

    for fold in folds_to_run:
        if any(r["fold"] == fold for r in results):
            print(f"Fold {fold} ja concluido, pulando", flush=True)
            continue

        set_seed(42 + fold)
        test_items = [it for it in items if it["fold"] == fold]
        remaining = [it for it in items if it["fold"] != fold]

        # Separa val a partir das imagens restantes (nao do fold de teste)
        remaining_image_ids = sorted(set(it["image_id"] for it in remaining))
        rng = random.Random(42 + fold)
        rng.shuffle(remaining_image_ids)
        n_val_images = int(len(remaining_image_ids) * args.val_fraction_of_train)
        val_image_ids = set(remaining_image_ids[:n_val_images])

        train_items = [it for it in remaining if it["image_id"] not in val_image_ids]
        val_items = [it for it in remaining if it["image_id"] in val_image_ids]

        print(f"\n[fold {fold}] train={len(train_items)} val={len(val_items)} test={len(test_items)} crops", flush=True)

        test_acc, test_f1, test_qwk, best_state = train_fold(train_items, val_items, test_items, device, args)
        print(f"[fold {fold}] Acc={test_acc:.4f} F1={test_f1:.4f} QWK={test_qwk:.4f}", flush=True)

        if args.save_models_dir:
            os.makedirs(args.save_models_dir, exist_ok=True)
            torch.save(best_state, os.path.join(args.save_models_dir, f"fold{fold}.pth"))

        results = [r for r in results if r["fold"] != fold]
        results.append({"fold": fold, "accuracy": test_acc, "f1_weighted": test_f1, "qwk": test_qwk})
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    if len(results) == args.k:
        accs = [r["accuracy"] for r in results]
        f1s = [r["f1_weighted"] for r in results]
        qwks = [r["qwk"] for r in results]
        print("\n" + "=" * 50, flush=True)
        print(f"K-FOLD ({args.k} folds) - RESUMO", flush=True)
        print("=" * 50, flush=True)
        print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
        print(f"F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}", flush=True)
        print(f"QWK      : {np.mean(qwks):.4f} ± {np.std(qwks):.4f}", flush=True)


if __name__ == "__main__":
    main()
