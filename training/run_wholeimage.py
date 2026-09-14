"""
Baseline "imagem inteira" (sem recorte por regiao): treina o mesmo backbone
ResNet18 (com o mesmo protocolo de treino -- augmentation, mixup, class
weights, early stopping) usado no pipeline principal, mas classificando a
FOTO INTEIRA em vez de um recorte anatomico unico.

Objetivo: isolar o efeito de "contexto visual completo do rosto" vs "recorte
pequeno de uma unica regiao" na acuracia -- para explicar honestamente por
que um classificador de imagem inteira (ex: modelo do Roboflow) tende a
reportar acuracia mais alta do que o nosso pipeline region-aware, sem que
isso signifique que a arquitetura region-aware esteja "errada": sao tarefas
de dificuldade diferente.

Usa exatamente a mesma divisao treino/val/teste (por imagem) do pipeline de
recorte por regiao -- ver utils/prepare_wholeimage_data.py.

Uso:
    cd training
    python utils/prepare_wholeimage_data.py --acne04-dir ../acne_1024
    python run_wholeimage.py --seed 42 --mixup-alpha 0.2
"""
import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets

from run_ablation import (
    ResNet18NoEmbed, build_transforms, train_model, set_seed,
)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WholeImageDataset(torch.utils.data.Dataset):
    """Wrapper fino sobre ImageFolder que devolve um region_idx dummy (0),
    so para reaproveitar train_one_epoch/evaluate/train_model sem alteracao."""

    def __init__(self, root, transform=None):
        self.ds = datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label, torch.tensor(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/wholeimage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--output", default="results/wholeimage_results.json")
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa-epochs", type=int, default=10)
    args = parser.parse_args()

    device = get_device(args.force_cpu)
    print(f"Device: {device}", flush=True)

    set_seed(args.seed)
    train_tf, val_tf = build_transforms(strong_aug=args.strong_aug)

    train_ds = WholeImageDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_ds = WholeImageDataset(os.path.join(args.data_dir, "val"), val_tf)
    test_ds = WholeImageDataset(os.path.join(args.data_dir, "test"), val_tf)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} imagens (inteiras)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_labels_train = [lbl for _, lbl in train_ds.ds.samples]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(all_labels_train), y=all_labels_train),
        dtype=torch.float,
    )

    model = ResNet18NoEmbed().to(device)
    acc, f1, qwk, best_state = train_model(
        model, train_loader, val_loader, test_loader, class_weights, device,
        num_epochs=args.epochs, weight_decay=args.weight_decay,
        loss_type=args.loss, mixup_alpha=args.mixup_alpha,
        swa=args.swa, swa_epochs=args.swa_epochs,
    )

    print(f"\n[seed={args.seed}] whole_image (sem recorte de regiao)", flush=True)
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

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
