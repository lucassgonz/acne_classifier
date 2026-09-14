"""
Fusao multi-regiao: em vez de classificar UM recorte anatomico por vez
(pipeline original), agrupa os ate 5 recortes (testa/queixo/nariz/bochecha
esq/dir) do MESMO rosto e faz UMA predicao de gravidade por imagem,
combinando as features de todas as regioes disponiveis.

Motivacao: o baseline de imagem inteira (data/wholeimage) mostrou que ver
mais contexto facial de uma vez eleva muito a acuracia (71.8% vs ~50% do
pipeline de crop unico). A fusao multi-regiao tenta recuperar parte desse
contexto (varias regioes da mesma pessoa) sem abrir mao da identidade de
regiao (mantida via embedding), que e a base da contribuicao do artigo.

Como a deteccao/qualidade de crop falha para varias imagens, a maioria dos
individuos NAO tem as 5 regioes completas (so ~6-8% tem as 5). Em vez de
descartar o resto, cada regiao ausente e substituida por um vetor
"placeholder" aprendido, e o pooling final e uma media mascarada apenas
sobre as regioes de fato presentes.

Uso:
    cd training
    python run_multiregion_fusion.py --seed 42 --mixup-alpha 0.2
"""
import argparse
import gc
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from run_ablation import _load_backbone, BACKBONE_OUT_FEATURES, build_transforms, REGIONS, CLASS_NAMES


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


class MultiRegionDataset(Dataset):
    """Agrupa os crops de regiao por imagem original (mesmo nome de arquivo
    entre as pastas de regiao = mesma pessoa/foto)."""

    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        by_image = defaultdict(dict)
        for region in REGIONS:
            for cls in ["0", "1", "2", "3"]:
                d = os.path.join(base_dir, region, split, cls)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        by_image[(fn, cls)][region] = os.path.join(d, fn)

        self.items = []
        for (fn, cls), region_paths in by_image.items():
            self.items.append((region_paths, int(cls)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        region_paths, label = self.items[idx]
        imgs = []
        mask = torch.zeros(len(REGIONS))
        for i, region in enumerate(REGIONS):
            if region in region_paths:
                img = Image.open(region_paths[region]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
                mask[i] = 1.0
            else:
                imgs.append(torch.zeros(3, 224, 224))
        return torch.stack(imgs), mask, label


class MultiRegionFusionNet(nn.Module):
    def __init__(self, num_classes=4, num_regions=5, pretrained_backbone_path=None, arch="resnet18"):
        super().__init__()
        self.backbone = _load_backbone(pretrained_backbone_path, arch=arch)
        in_features = BACKBONE_OUT_FEATURES[arch]
        self.num_regions = num_regions
        self.region_embed = nn.Embedding(num_regions, 64)
        self.missing_placeholder = nn.Parameter(torch.zeros(in_features))
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

    def forward(self, x, mask):
        # x: [B, R, C, H, W]   mask: [B, R]
        B, R, C, H, W = x.shape
        feat = self.backbone(x.view(B * R, C, H, W)).flatten(1).view(B, R, -1)

        m = mask.unsqueeze(-1)  # [B, R, 1]
        placeholder = self.missing_placeholder.view(1, 1, -1).expand(B, R, -1)
        feat = feat * m + placeholder * (1 - m)

        region_idx = torch.arange(R, device=x.device).unsqueeze(0).expand(B, R)
        emb = self.region_embed(region_idx)
        combined = torch.cat([feat, emb], dim=-1)  # [B, R, in_features+64]

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (combined * m).sum(dim=1) / denom  # media mascarada
        return self.classifier(pooled)


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(imgs.size(0), device=device)
            mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
            # so conta a regiao como valida se estiver presente nas DUAS
            # amostras -- caso contrario a mistura seria conteudo real +
            # preto (regiao ausente), mas marcada como dado limpo (bug)
            mixed_masks = masks * masks[perm]
            out = model(mixed_imgs, mixed_masks)
            loss = lam * criterion(out, labels) + (1 - lam) * criterion(out, labels[perm])
        else:
            out = model(imgs, masks)
            loss = criterion(out, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, masks, labels in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs, masks)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return acc, f1, qwk


def train_model(model, train_loader, val_loader, test_loader, class_weights, device,
                 num_epochs=60, patience=20, weight_decay=1e-4, mixup_alpha=0.0):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=mixup_alpha)
        val_acc, val_f1, val_qwk = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(f"  epoch {epoch}: loss={train_loss:.4f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_acc, test_f1, test_qwk = evaluate(model, test_loader, device)
    return test_acc, test_f1, test_qwk, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--output", default="results/multiregion_fusion.json")
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    args = parser.parse_args()

    device = get_device(args.force_cpu)
    print(f"Device: {device}", flush=True)

    set_seed(args.seed)
    train_tf, val_tf = build_transforms()

    train_ds = MultiRegionDataset(args.data_dir, "train", train_tf)
    val_ds = MultiRegionDataset(args.data_dir, "val", val_tf)
    test_ds = MultiRegionDataset(args.data_dir, "test", val_tf)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} imagens (multi-regiao)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_labels_train = [lbl for _, lbl in train_ds.items]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(all_labels_train), y=all_labels_train),
        dtype=torch.float,
    )

    model = MultiRegionFusionNet(pretrained_backbone_path=args.pretrained_backbone, arch=args.arch).to(device)
    acc, f1, qwk, best_state = train_model(
        model, train_loader, val_loader, test_loader, class_weights, device,
        num_epochs=args.epochs, weight_decay=args.weight_decay, mixup_alpha=args.mixup_alpha,
    )

    print(f"\n[seed={args.seed}] multiregion_fusion", flush=True)
    print(f"  Test Acc={acc:.4f}  F1={f1:.4f}  QWK={qwk:.4f}", flush=True)

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(best_state, args.save_model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = {"seed": args.seed, "accuracy": acc, "f1_weighted": f1, "qwk": qwk}
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

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
