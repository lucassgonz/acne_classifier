"""
Ablation study: ResNet-18 com embedding anatômico vs. sem embedding.

Executa N seeds para cada configuração, mantendo exatamente o mesmo
protocolo de dados, augmentation, otimizador e scheduler.

Uso:
    cd training
    python run_ablation.py --data-dir ../../data/final --seeds 3
"""
import argparse
import gc
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
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
# Modelos
# ---------------------------------------------------------------------------

def _load_backbone(pretrained_backbone_path):
    """
    Cria um resnet18. Se pretrained_backbone_path for informado, carrega o
    backbone pré-treinado (ex: SCIN, que já parte de ImageNet). Caso
    contrário, usa pesos ImageNet como ponto de partida padrão — treinar do
    zero (weights=None) é inadequado para um dataset do tamanho do ACNE04.
    """
    if pretrained_backbone_path is not None:
        base = models.resnet18(weights=None)
        state = torch.load(pretrained_backbone_path, map_location="cpu")
        base.load_state_dict(state, strict=False)
    else:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return nn.Sequential(*list(base.children())[:-1])


class ResNet18WithEmbed(nn.Module):
    """Modelo original do paper: ResNet-18 + embedding anatômico (64-d)."""

    def __init__(self, num_classes=4, num_regions=5, pretrained_backbone_path=None):
        super().__init__()
        self.backbone = _load_backbone(pretrained_backbone_path)
        in_features = 512
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
        feat = self.backbone(x).flatten(1)
        emb = self.region_embed(region_idx.long()).squeeze(1)
        return self.classifier(torch.cat([feat, emb], dim=1))


class ResNet18NoEmbed(nn.Module):
    """Baseline de ablação: mesma arquitetura sem embedding de região."""

    def __init__(self, num_classes=4, pretrained_backbone_path=None):
        super().__init__()
        self.backbone = _load_backbone(pretrained_backbone_path)
        in_features = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, region_idx=None):
        feat = self.backbone(x).flatten(1)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
CLASS_NAMES = ["Level_0", "Level_1", "Level_2", "Level_3"]


class AcneDataset(Dataset):
    def __init__(self, base_dir, splits, transform=None):
        self.samples = []
        self.transform = transform
        for region in REGIONS:
            for split in splits:
                split_dir = os.path.join(base_dir, region, split)
                if not os.path.isdir(split_dir):
                    continue
                ds = datasets.ImageFolder(split_dir)
                for path, label in ds.samples:
                    self.samples.append((path, label, REGIONS.index(region)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, region_idx = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, torch.tensor(region_idx)


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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) com pesos por classe. Reduz a contribuicao
    de exemplos faceis (bem classificados) na loss, focando o treino nas
    classes minoritarias/dificeis -- relevante aqui pela forte imbalance
    entre niveis de gravidade (ex: nivel 3 tem poucas amostras).
    """

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Treino / Avaliação
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, regions in loader:
        imgs, labels, regions = imgs.to(device), labels.to(device), regions.to(device)
        optimizer.zero_grad()
        out = model(imgs, regions)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, labels, regions = imgs.to(device), labels.to(device), regions.to(device)
        out = model(imgs, regions)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return acc, f1, qwk


def train_model(model, train_loader, val_loader, test_loader,
                class_weights, device, num_epochs=40, patience=10, weight_decay=1e-4,
                loss_type="weighted_ce"):
    if loss_type == "focal":
        criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, val_qwk = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    test_acc, test_f1, test_qwk = evaluate(model, test_loader, device)
    return test_acc, test_f1, test_qwk, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/final")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                        help="Roda apenas esta seed (ignora --seeds). Usado para isolar cada rodada em processo separado.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--only-config", choices=["with_embedding", "no_embedding"], default=None,
                        help="Roda apenas essa configuracao (pula a outra). Util para gerar checkpoints de um unico modelo.")
    parser.add_argument("--output", default="results/ablation_results.json")
    parser.add_argument("--pretrained-backbone", default=None,
                        help="Caminho para backbone pré-treinado (ex: models/backbone_scin_pretrained.pth)")
    parser.add_argument("--save-models-dir", default=None,
                        help="Se informado, salva o state_dict de cada modelo treinado (por seed/config) nesse diretorio, para uso posterior em ensemble/TTA.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms()

    train_ds = AcneDataset(args.data_dir, ["train"], train_tf)
    val_ds = AcneDataset(args.data_dir, ["val"], val_tf)
    test_ds = AcneDataset(args.data_dir, ["test"], val_tf)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} crops")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    all_labels_train = [s[1] for s in train_ds.samples]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(all_labels_train), y=all_labels_train),
        dtype=torch.float,
    )

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(42, 42 + args.seeds))

    # Continua um arquivo de resultados existente (util quando cada seed
    # roda em um processo separado via --seed)
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = {"with_embedding": [], "no_embedding": []}

    configs = [True, False]
    if args.only_config == "with_embedding":
        configs = [True]
    elif args.only_config == "no_embedding":
        configs = [False]

    for seed in seeds:
        for use_embed in configs:
            tag = "with_embedding" if use_embed else "no_embedding"

            if any(r["seed"] == seed for r in results[tag]):
                print(f"\n[seed={seed}] {tag} -- ja concluida, pulando")
                continue

            print(f"\n[seed={seed}] {tag}", flush=True)
            set_seed(seed)

            model = (
                ResNet18WithEmbed(pretrained_backbone_path=args.pretrained_backbone).to(device)
                if use_embed
                else ResNet18NoEmbed(pretrained_backbone_path=args.pretrained_backbone).to(device)
            )
            acc, f1, qwk, best_state = train_model(
                model, train_loader, val_loader, test_loader,
                class_weights, device,
                num_epochs=args.epochs,
                weight_decay=args.weight_decay,
                loss_type=args.loss,
            )
            results[tag].append({"seed": seed, "accuracy": acc, "f1_weighted": f1, "qwk": qwk})
            print(f"  Test Acc={acc:.4f}  F1={f1:.4f}  QWK={qwk:.4f}", flush=True)

            if args.save_models_dir:
                os.makedirs(args.save_models_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_models_dir, f"{tag}_seed{seed}.pth")
                torch.save(best_state, ckpt_path)

            # Libera memoria do modelo/otimizador antes da proxima rodada
            del model, best_state
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

            # Salva progresso parcial a cada rodada (protege contra queda de memoria/energia)
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("ABLATION STUDY - RESUMO")
    print("=" * 50)
    for tag, runs in results.items():
        accs = [r["accuracy"] for r in runs]
        f1s = [r["f1_weighted"] for r in runs]
        qwks = [r.get("qwk", float("nan")) for r in runs]
        print(f"\n{tag}:")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  QWK      : {np.mean(qwks):.4f} ± {np.std(qwks):.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {args.output}")


if __name__ == "__main__":
    main()
