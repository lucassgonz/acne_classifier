"""
Classificacao ORDINAL da gravidade de acne usando CORAL (Cao, Mirjalili &
Raschka, 2020 -- "Rank Consistent Ordinal Regression for Neural Networks").

Diferente da classificacao multi-classe padrao (usada em run_ablation.py),
aqui a gravidade 0-3 e tratada como um problema ORDENADO: em vez de uma
softmax de 4 classes independentes, o modelo aprende 3 classificadores
binarios "a gravidade e maior que k?" (k=0,1,2) que compartilham a mesma
representacao. Isso penaliza menos erros entre classes vizinhas (nivel 1
previsto como 2) do que erros distantes (nivel 0 previsto como 3),
capturando a estrutura ordinal real do problema.

Mesmo backbone e embedding de regiao do modelo proposto (run_ablation.py)
-- so a cabeca de classificacao e a loss mudam.

Uso:
    cd training
    python run_ordinal.py --data-dir ../data/final --seed 42 \
        --pretrained-backbone models/backbone_scin_pretrained_v2.pth \
        --mixup-alpha 0.2
"""
import argparse
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import models

from run_ablation import AcneDataset, REGIONS, CLASS_NAMES, build_transforms, _load_backbone


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
# Modelo com cabeca ordinal CORAL
# ---------------------------------------------------------------------------

class CoralResNet18WithEmbed(nn.Module):
    """
    CORAL com garantia de rank-consistency: os limiares (bias) sao
    parametrizados como uma sequencia estritamente decrescente
    (b_0 > b_1 > ... > b_{K-2}), obtida subtraindo incrementos sempre
    positivos (softplus) do primeiro limiar. Isso e o que garante que
    P(y>0) >= P(y>1) >= ... -- sem essa restricao, o CORAL "quebra" e as
    predicoes ficam inconsistentes (o que causou o resultado ruim da
    primeira tentativa).
    """

    def __init__(self, num_classes=4, num_regions=5, pretrained_backbone_path=None):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.backbone = _load_backbone(pretrained_backbone_path)
        in_features = 512
        self.region_embed = nn.Embedding(num_regions, 64)
        self.feature_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.coral_weight = nn.Linear(128, 1, bias=False)
        # bias_raw[0] = limiar inicial (livre); bias_raw[1:] = incrementos
        # (sempre positivos via softplus) subtraidos em cascata
        self.bias_raw = nn.Parameter(torch.zeros(self.num_thresholds).float())

    def ordered_bias(self):
        first = self.bias_raw[0:1]
        increments = F.softplus(self.bias_raw[1:])
        decreases = torch.cumsum(increments, dim=0)
        rest = first - decreases
        return torch.cat([first, rest], dim=0)

    def forward(self, x, region_idx):
        feat = self.backbone(x).flatten(1)
        emb = self.region_embed(region_idx.long()).squeeze(1)
        h = self.feature_head(torch.cat([feat, emb], dim=1))
        return self.coral_weight(h) + self.ordered_bias()  # [batch, num_classes-1]


# ---------------------------------------------------------------------------
# Loss e conversao de rotulos ordinais
# ---------------------------------------------------------------------------

def labels_to_levels(labels, num_classes):
    """labels [batch] (int 0..K-1) -> levels [batch, K-1] binario (y > k?)."""
    batch_size = labels.size(0)
    levels = torch.zeros(batch_size, num_classes - 1, device=labels.device)
    for k in range(num_classes - 1):
        levels[:, k] = (labels > k).float()
    return levels


def coral_loss(logits, levels, class_weight_per_threshold=None):
    """Loss do CORAL (Cao et al., 2020)."""
    term = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels)
    if class_weight_per_threshold is not None:
        term = term * class_weight_per_threshold
    return -torch.sum(term, dim=1).mean()


def coral_predict(logits):
    """logits [batch, K-1] -> rotulo previsto [batch] (soma de limiares excedidos)."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


# ---------------------------------------------------------------------------
# Treino / avaliacao
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, regions = imgs.to(device), regions.to(device)
        logits = model(imgs, regions)
        preds = coral_predict(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return acc, f1, qwk


def train_one_epoch(model, loader, optimizer, device, num_classes, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    for imgs, labels, regions in loader:
        imgs, labels, regions = imgs.to(device), labels.to(device), regions.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(imgs.size(0), device=device)
            mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
            logits = model(mixed_imgs, regions)
            levels_a = labels_to_levels(labels, num_classes)
            levels_b = labels_to_levels(labels[perm], num_classes)
            loss = lam * coral_loss(logits, levels_a) + (1 - lam) * coral_loss(logits, levels_b)
        else:
            logits = model(imgs, regions)
            levels = labels_to_levels(labels, num_classes)
            loss = coral_loss(logits, levels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, val_loader, test_loader, device,
                num_classes=4, num_epochs=40, patience=10, weight_decay=1e-4, mixup_alpha=0.0):
    # bias_raw define a separacao entre os niveis de gravidade -- weight
    # decay puxaria esses valores para zero e colapsaria os limiares,
    # entao ele (e demais bias/BatchNorm) fica de fora da regularizacao.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if "bias_raw" in name or "bias" in name or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, num_classes, mixup_alpha=mixup_alpha)
        val_acc, val_f1, val_qwk = evaluate(model, val_loader, device, num_classes)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_acc, test_f1, test_qwk = evaluate(model, test_loader, device, num_classes)
    return test_acc, test_f1, test_qwk, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--output", default="results/ordinal_results.json")
    args = parser.parse_args()

    device = get_device()
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

    num_classes = len(CLASS_NAMES)
    model = CoralResNet18WithEmbed(
        num_classes=num_classes, pretrained_backbone_path=args.pretrained_backbone
    ).to(device)

    acc, f1, qwk, best_state = train_model(
        model, train_loader, val_loader, test_loader, device,
        num_classes=num_classes, num_epochs=args.epochs,
        weight_decay=args.weight_decay, mixup_alpha=args.mixup_alpha,
    )

    print(f"\nResultado (seed={args.seed}, CORAL ordinal):", flush=True)
    print(f"  Test Acc={acc:.4f}  F1={f1:.4f}  QWK={qwk:.4f}", flush=True)

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(best_state, args.save_model)
        print(f"Modelo salvo em {args.save_model}", flush=True)

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


if __name__ == "__main__":
    main()
