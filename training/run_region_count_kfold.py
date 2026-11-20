"""
Region-Aware ResNet-18 com supervisao de contagem de lesoes POR REGIAO.

Mesmo protocolo de run_kfold_clean.py (folds de results/kfold_splits_v3.json,
Adam lr 1e-4, weighted CE, mixup 0.2, early stopping pela acuracia de validacao,
40 epocas, patience 10), com uma cabeca auxiliar que preve a distribuicao do
numero de lesoes anotadas dentro do proprio recorte (results/region_lesion_counts.json,
gerado por utils/build_region_lesion_counts.py). O alvo e uma distribuicao
gaussiana discreta centrada na contagem (sigma fixo), com loss KL, como na
formulacao de contagem do JAGC; recortes sem contagem conhecida (-1) nao
recebem essa loss.

--count-weight 0 desliga a cabeca de contagem (mesmo codigo, para a comparacao
controlada). --no-embed remove o embedding anatomico.

Uso:
    cd training
    python run_region_count_kfold.py --name rc_with --fold 0
    python run_region_count_kfold.py --name rc_without --fold 0 --no-embed
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from run_ablation import BACKBONE_OUT_FEATURES, _load_backbone, build_optimizer, build_transforms, get_device, set_seed
from run_kfold_clean import gather, index_crops
from run_ablation import REGIONS


class RegionCountNet(nn.Module):
    def __init__(self, num_bins, use_embed=True, pretrained_backbone_path=None, num_classes=4, num_regions=5):
        super().__init__()
        self.backbone = _load_backbone(pretrained_backbone_path, arch="resnet18")
        feat_dim = BACKBONE_OUT_FEATURES["resnet18"]
        self.use_embed = use_embed
        emb_dim = 64 if use_embed else 0
        if use_embed:
            self.region_embed = nn.Embedding(num_regions, 64)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim + emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.count_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim + emb_dim, num_bins))

    def forward(self, x, region_idx, with_count=False):
        h = self.backbone(x).flatten(1)
        if self.use_embed:
            h = torch.cat([h, self.region_embed(region_idx.long()).view(-1, 64)], dim=1)
        logits = self.classifier(h)
        if with_count:
            return logits, self.count_head(h)
        return logits


def gaussian_targets(counts, num_bins, sigma):
    bins = torch.arange(num_bins, dtype=torch.float, device=counts.device)
    c = counts.clamp(min=0, max=num_bins - 1).float().unsqueeze(1)
    d = torch.exp(-((bins - c) ** 2) / (2 * sigma ** 2))
    return d / d.sum(dim=1, keepdim=True)


class CountDataset(Dataset):
    def __init__(self, samples, counts, transform):
        self.samples = samples
        self.counts = counts
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, path, label, region = self.samples[idx]
        count = self.counts.get(f"{REGIONS[region]}/{name}", {"count": -1})["count"]
        img = self.transform(Image.open(path).convert("RGB")).numpy()
        return img, np.int64(label), np.int64(region), np.int64(count)


def numpy_collate(batch):
    """Batch montado em numpy no worker; tensores so no processo principal (no Windows,
    Python 3.15 + torch 2.14, o envio de tensores entre processos derruba os workers)."""
    return tuple(np.stack([b[k] for b in batch]) for k in range(len(batch[0])))


class TensorLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield tuple(torch.from_numpy(b) for b in batch)

    def __len__(self):
        return len(self.loader)


def make_loader(ds, batch_size, shuffle, workers, drop_last=False):
    return TensorLoader(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                   num_workers=workers, collate_fn=numpy_collate, persistent_workers=workers > 0))


def count_kl(count_logits, counts, num_bins, sigma):
    mask = (counts >= 0).float()
    if mask.sum() == 0:
        return count_logits.sum() * 0.0
    target = gaussian_targets(counts, num_bins, sigma)
    kl = F.kl_div(F.log_softmax(count_logits, dim=1), target, reduction="none").sum(dim=1)
    return (kl * mask).sum() / mask.sum()


def train_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    for imgs, labels, regions, counts in loader:
        imgs, labels, regions, counts = imgs.to(device), labels.to(device), regions.to(device), counts.to(device)
        optimizer.zero_grad()
        if args.mixup_alpha > 0:
            lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            perm = torch.randperm(imgs.size(0), device=device)
            logits, cl = model(lam * imgs + (1 - lam) * imgs[perm], regions, with_count=True)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
            if args.count_weight > 0:
                loss = loss + args.count_weight * (
                    lam * count_kl(cl, counts, args.count_bins, args.count_sigma)
                    + (1 - lam) * count_kl(cl, counts[perm], args.count_bins, args.count_sigma))
        else:
            logits, cl = model(imgs, regions, with_count=True)
            loss = criterion(logits, labels)
            if args.count_weight > 0:
                loss = loss + args.count_weight * count_kl(cl, counts, args.count_bins, args.count_sigma)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, exp_counts = [], []
    for imgs, _, regions, _ in loader:
        logits, cl = model(imgs.to(device), regions.to(device), with_count=True)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        pc = F.softmax(cl, dim=1)
        exp_counts.append((pc * torch.arange(pc.size(1), device=pc.device)).sum(1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(exp_counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--splits", default="results/kfold_splits_v3.json")
    p.add_argument("--data-dir", default="../data/final_v3")
    p.add_argument("--counts", default="results/region_lesion_counts.json")
    p.add_argument("--output-dir", default="results/kfold")
    p.add_argument("--pretrained-backbone", default=None)
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--count-weight", type=float, default=1.0)
    p.add_argument("--count-bins", type=int, default=31)
    p.add_argument("--count-sigma", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.name}] fold {args.fold} ja concluido, pulando")
        return

    fold = json.load(open(args.splits))["folds"][args.fold]
    counts = json.load(open(args.counts))
    index = index_crops(args.data_dir)
    train_s, val_s, test_s = (gather(index, fold[k]) for k in ("train", "val", "test"))
    print(f"[{args.name}] fold {args.fold}: crops train={len(train_s)} val={len(val_s)} test={len(test_s)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=False, size=224)
    train_loader = make_loader(CountDataset(train_s, counts, train_tf), args.batch_size, True, args.workers, drop_last=True)
    val_loader = make_loader(CountDataset(val_s, counts, val_tf), args.batch_size, False, args.workers)
    test_loader = make_loader(CountDataset(test_s, counts, val_tf), args.batch_size, False, args.workers)

    y_train = [s[2] for s in train_s]
    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train), dtype=torch.float)

    set_seed(args.seed)
    model = RegionCountNet(args.count_bins, use_embed=not args.no_embed,
                           pretrained_backbone_path=args.pretrained_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(model, "adam", args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    start = time.time()
    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, args)
        vp, _ = predict(model, val_loader, device)
        val_acc = accuracy_score([s[2] for s in val_s], vp.argmax(1))
        scheduler.step(val_acc)
        print(f"[{args.name}] fold {args.fold} epoch {epoch}: val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    probs, exp_counts = predict(model, test_loader, device)
    labels = np.array([s[2] for s in test_s])
    pred = probs.argmax(1)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(labels, pred, weights="quadratic")
    minutes = (time.time() - start) / 60
    print(f"[{args.name}] fold {args.fold}: Acc={acc:.4f} F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)
    np.savez(
        npz_path,
        names=np.array([s[0] for s in test_s]),
        regions=np.array([s[3] for s in test_s]),
        labels=labels,
        probs=probs,
        exp_counts=exp_counts,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
