"""
Teste do modelo proposto usando ViT-B/16 pre-treinado (ImageNet) como
backbone alternativo ao ResNet18, mantendo o embedding de regiao e o
resto do protocolo de treino identicos (mesmo dado, mesma augmentation,
mesmo Mixup opcional).

ATENCAO: ViT-B/16 tem ~86M parametros (vs ~11.7M do ResNet18) -- espera-se
uso de memoria e tempo de treino/inferencia bem maiores. Batch size menor
por padrao para reduzir risco de estourar memoria.

Uso:
    cd training
    python run_vit.py --data-dir ../data/final --seed 42 --mixup-alpha 0.2
"""
import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import models

from run_ablation import (
    AcneDataset, build_transforms, train_one_epoch, evaluate, FocalLoss,
)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ViTWithEmbed(nn.Module):
    def __init__(self, num_classes=4, num_regions=5):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()
        self.backbone = vit
        in_features = 768

        self.region_embed = nn.Embedding(num_regions, 64)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features + 64, 256),
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
        feat = self.backbone(x)
        emb = self.region_embed(region_idx.long()).squeeze(1)
        return self.classifier(torch.cat([feat, emb], dim=1))


def train_model(model, train_loader, val_loader, test_loader, class_weights, device,
                num_epochs=40, patience=10, weight_decay=1e-4, mixup_alpha=0.0):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=mixup_alpha)
        epoch_times.append(time.time() - t0)
        val_acc, val_f1, val_qwk = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(f"  epoch {epoch}: val_acc={val_acc:.4f}  ({epoch_times[-1]:.1f}s)", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)

    t0 = time.time()
    test_acc, test_f1, test_qwk = evaluate(model, test_loader, device)
    inference_time = time.time() - t0

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    return test_acc, test_f1, test_qwk, best_state, avg_epoch_time, inference_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--output", default="results/vit_results.json")
    args = parser.parse_args()

    device = get_device(args.force_cpu)
    print(f"Device: {device}", flush=True)

    set_seed(args.seed)
    train_tf, val_tf = build_transforms()

    train_ds = AcneDataset(args.data_dir, ["train"], train_tf)
    val_ds = AcneDataset(args.data_dir, ["val"], val_tf)
    test_ds = AcneDataset(args.data_dir, ["test"], val_tf)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} crops", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_labels_train = [s[1] for s in train_ds.samples]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(all_labels_train), y=all_labels_train),
        dtype=torch.float,
    )

    print("Carregando ViT-B/16 pre-treinado...", flush=True)
    model = ViTWithEmbed().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros totais: {n_params/1e6:.1f}M", flush=True)

    acc, f1, qwk, best_state, avg_epoch_time, inference_time = train_model(
        model, train_loader, val_loader, test_loader, class_weights, device,
        num_epochs=args.epochs, weight_decay=args.weight_decay, mixup_alpha=args.mixup_alpha,
    )

    print(f"\nResultado (seed={args.seed}, ViT-B/16):", flush=True)
    print(f"  Test Acc={acc:.4f}  F1={f1:.4f}  QWK={qwk:.4f}", flush=True)
    print(f"  Tempo medio/epoca: {avg_epoch_time:.1f}s", flush=True)
    print(f"  Tempo de inferencia no teste ({len(test_ds)} crops): {inference_time:.2f}s ({inference_time/len(test_ds)*1000:.1f}ms/crop)", flush=True)

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(best_state, args.save_model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = {
        "seed": args.seed, "accuracy": acc, "f1_weighted": f1, "qwk": qwk,
        "avg_epoch_time_s": avg_epoch_time, "inference_time_s": inference_time,
        "n_params": n_params,
    }
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results = [r for r in all_results if r["seed"] != args.seed]
    all_results.append(result)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Resultados salvos em {args.output}", flush=True)


if __name__ == "__main__":
    main()
