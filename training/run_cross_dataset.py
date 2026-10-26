"""
Ablacao de generalizacao entre fontes (item prioritario da revisao):
treina no Acne04 (acne_1024/, 1406 fotos) inteiro e testa no AcneLevel
TRATADO (data/acnelevel_augmented/, cor/brilho/posicao alterados por
utils/augment_acnelevel.py -- ver esse script para o motivo), e vice-versa.

Dataset/ e acne_1024/ compartilham ~1406 das mesmas fotos (mesmo acervo,
resolucao diferente); por isso usamos a versao TRATADA do AcneLevel aqui,
nao Dataset/ bruto -- rodar utils/augment_acnelevel.py antes.

Como e um unico split treino/teste por fonte (nao k-fold), roda rapido:
2 execucoes no total (uma por direcao), nao 2x5.

Uso:
    cd training
    python utils/augment_acnelevel.py
    python run_cross_dataset.py --train-source acne04
    python run_cross_dataset.py --train-source acnelevel
"""
import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from run_ablation import ResNet18NoEmbed, build_transforms, get_device, set_seed, train_model
from run_wholeimage_kfold import (
    WholeImageSplitDataset, build_tta_transforms, predict, predict_tta,
)


def index_acne04():
    samples = []
    for level in range(4):
        for p in glob.glob(f"../acne_1024/levle{level}_*.jpg"):
            samples.append((os.path.basename(p), p, level))
    return samples


def index_acnelevel_augmented(root="../data/acnelevel_augmented"):
    samples = []
    for level in range(4):
        for p in glob.glob(os.path.join(root, str(level), "*.jpg")):
            samples.append((os.path.basename(p), p, level))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-source", required=True, choices=["acne04", "acnelevel"])
    p.add_argument("--output-dir", default="results/cross_dataset")
    p.add_argument("--pretrained-backbone", default="models/backbone_scin_pretrained_v2.pth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--swa", action="store_true", default=True)
    p.add_argument("--swa-epochs", type=int, default=10)
    p.add_argument("--tta-views", type=int, default=7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_{args.train_source}.npz"
    if out_path.exists():
        print(f"[{args.train_source}] ja concluido, pulando")
        return

    acne04_samples = index_acne04()
    acnelevel_samples = index_acnelevel_augmented()
    print(f"Acne04: {len(acne04_samples)} imagens | AcneLevel (tratado): {len(acnelevel_samples)} imagens", flush=True)

    train_pool, test_s = (acne04_samples, acnelevel_samples) if args.train_source == "acne04" else (acnelevel_samples, acne04_samples)

    rng = random.Random(args.seed)
    train_pool = train_pool[:]
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * args.val_frac)
    val_s = train_pool[:n_val]
    train_s = train_pool[n_val:]
    print(f"[{args.train_source}] train={len(train_s)} val={len(val_s)} test(other source)={len(test_s)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=False, size=224)
    train_loader = DataLoader(WholeImageSplitDataset(train_s, train_tf), batch_size=args.batch_size,
                               shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(WholeImageSplitDataset(val_s, val_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(WholeImageSplitDataset(test_s, val_tf), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    y_train = [s[2] for s in train_s]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train), dtype=torch.float
    )

    set_seed(args.seed)
    model = ResNet18NoEmbed(pretrained_backbone_path=args.pretrained_backbone).to(device)

    start = time.time()
    acc, f1, qwk, best_state = train_model(
        model, train_loader, val_loader, test_loader, class_weights, device,
        num_epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay,
        loss_type="weighted_ce", mixup_alpha=args.mixup_alpha, lr=args.lr,
        swa=args.swa, swa_epochs=args.swa_epochs,
    )
    minutes = (time.time() - start) / 60
    print(f"[{args.train_source}] single-view: Acc={acc:.4f} F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)

    if args.tta_views > 1:
        tta_views = build_tta_transforms(args.tta_views)
        probs = predict_tta(model, test_s, tta_views, device, args.batch_size)
    else:
        probs = predict(model, test_loader, device)

    np.savez(
        out_path,
        names=np.array([s[0] for s in test_s]),
        labels=np.array([s[2] for s in test_s]),
        probs=probs,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    with open(out_dir / f"config_{args.train_source}.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
