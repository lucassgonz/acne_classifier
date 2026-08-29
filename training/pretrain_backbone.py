"""
Pre-treino supervisionado do backbone ResNet18 no SCIN (classificacao de
condicao dermatologica, ~18 classes) antes do fine-tuning no ACNE04.

Os 4 niveis de gravidade de acne usados no treino final vem exclusivamente
do ACNE04 — o SCIN nao tem gravidade, serve apenas para o backbone
aprender representacoes de pele mais robustas a partir de um corpus maior
e mais diverso antes da especializacao na tarefa de gravidade de acne.

Uso:
    cd training
    python pretrain_backbone.py --data-dir ../../data/scin_pretrain --epochs 15
    # gera: models/backbone_scin_pretrained.pth
"""
import argparse
import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/scin_pretrain")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", default="models/backbone_scin_pretrained.pth")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Forca uso de CPU em vez de MPS (mais lento, porem com uso de memoria mais previsivel)")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms()

    # Duas instancias separadas (uma por transform) sobre os MESMOS indices,
    # para train e val nao compartilharem o mesmo objeto Dataset (isso faria
    # a transform de val sobrescrever a de train, removendo o augmentation).
    base_ds = datasets.ImageFolder(args.data_dir)
    num_classes = len(base_ds.classes)
    print(f"Classes ({num_classes}): {base_ds.classes}")
    print(f"Total de imagens: {len(base_ds)}")

    n_val = int(len(base_ds) * args.val_ratio)
    n_train = len(base_ds) - n_val
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(base_ds)), [n_train, n_val], generator=generator
    )

    train_full = datasets.ImageFolder(args.data_dir, transform=train_tf)
    val_full = datasets.ImageFolder(args.data_dir, transform=val_tf)
    train_ds = torch.utils.data.Subset(train_full, train_idx.indices)
    val_ds = torch.utils.data.Subset(val_full, val_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Backbone com pesos ImageNet, cabeça trocada para o numero de classes do SCIN
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_backbone_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if device.type == "mps" and batch_idx % 50 == 49:
                torch.mps.empty_cache()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)

        # Evita acumulo de memoria do alocador MPS ao longo das epocas
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"Epoch {epoch}/{args.epochs}  loss={running_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # Salva apenas as camadas convolucionais (sem a fc do SCIN),
            # compativeis com OptimizedMultiRegionResNet18.resnet
            backbone_only = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
            best_backbone_state = {k: v.cpu().clone() for k, v in backbone_only.items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(best_backbone_state, args.output)
    print(f"\nMelhor val_acc no SCIN: {best_val_acc:.4f}")
    print(f"Backbone salvo em: {args.output}")
    print("Use este arquivo com --pretrained-backbone no treino do ACNE04.")


if __name__ == "__main__":
    main()
