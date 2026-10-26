"""
Modelo ESPECIALISTA por regiao: em vez do backbone unico compartilhado com
embedding de regiao (pipeline principal), treina UM ResNet-18 proprio por
regiao anatomica (sem embedding -- a especializacao vem dos pesos, nao de um
vetor de condicionamento), nos mesmos folds por grupo do k-fold limpo.

Usa 5x mais compute (5 regioes x 5 folds em vez de 1x5), mas testa a
hipotese de que pesos dedicados por regiao capturam melhor a textura/padrao
de lesao especifico daquela area do rosto do que um backbone compartilhado.

Uso (piloto, so fold 0, todas as regioes):
    cd training
    for r in forehead chin nose left_cheek right_cheek; do
        python run_region_specialist_kfold.py --region $r --fold 0 \
            --pretrained-backbone models/backbone_scin_pretrained_v2.pth --mixup-alpha 0.2
    done
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from run_ablation import REGIONS, ResNet18NoEmbed, build_transforms, get_device, set_seed, train_model


class SampleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, torch.tensor(-1)


def index_region_crops(data_dir, region):
    index = defaultdict(list)
    for split in ("train", "val", "test"):
        for cls in range(4):
            d = os.path.join(data_dir, region, split, str(cls))
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    index[fn].append((fn, os.path.join(d, fn), cls))
    return index


def gather(index, names):
    out = []
    for n in names:
        out.extend(index.get(n, []))
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for imgs, _, _ in loader:
        probs.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=REGIONS)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--splits", default="results/kfold_splits.json")
    p.add_argument("--data-dir", default="../data/final")
    p.add_argument("--output-dir", default="results/kfold_specialist")
    p.add_argument("--pretrained-backbone", default=None)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    p.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "focal", "ordinal_ls"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--ordinal-tau", type=float, default=0.7)
    p.add_argument("--head-lr-mult", type=float, default=1.0)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--strong-aug", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--swa", action="store_true")
    p.add_argument("--swa-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.region)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.region}] fold {args.fold} ja concluido, pulando")
        return

    with open(args.splits) as f:
        fold = json.load(f)["folds"][args.fold]
    index = index_region_crops(args.data_dir, args.region)
    train_s, val_s, test_s = (gather(index, fold[k]) for k in ("train", "val", "test"))
    print(f"[{args.region}] fold {args.fold}: crops train={len(train_s)} val={len(val_s)} test={len(test_s)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=args.strong_aug, size=args.img_size)
    train_loader = DataLoader(SampleDataset(train_s, train_tf), batch_size=args.batch_size,
                               shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(SampleDataset(val_s, val_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(SampleDataset(test_s, val_tf), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    y_train = [s[2] for s in train_s]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train), dtype=torch.float
    )

    set_seed(args.seed)
    model = ResNet18NoEmbed(pretrained_backbone_path=args.pretrained_backbone, arch=args.arch).to(device)

    start = time.time()
    acc, f1, qwk, best_state = train_model(
        model, train_loader, val_loader, test_loader, class_weights, device,
        num_epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay,
        loss_type=args.loss, mixup_alpha=args.mixup_alpha, lr=args.lr,
        swa=args.swa, swa_epochs=args.swa_epochs,
        optimizer_name=args.optimizer, scheduler_name=args.scheduler,
        label_smoothing=args.label_smoothing, head_lr_mult=args.head_lr_mult, ordinal_tau=args.ordinal_tau,
    )
    minutes = (time.time() - start) / 60
    print(f"[{args.region}] fold {args.fold}: Acc={acc:.4f} F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)

    probs = predict(model, test_loader, device)
    np.savez(
        npz_path,
        names=np.array([s[0] for s in test_s]),
        labels=np.array([s[2] for s in test_s]),
        probs=probs,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
