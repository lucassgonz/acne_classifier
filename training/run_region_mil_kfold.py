"""
Region-Aware ResNet-18 treinado com o rotulo da FOTO (multiple-instance learning).

Cada amostra e uma foto com todos os seus recortes de regiao validos (1 a 5). O mesmo
backbone (inicializacao SCIN) extrai as features de cada recorte, concatenadas ao
embedding anatomico de 64 dimensoes da regiao, como no modelo por recorte. Um pooling
por atencao (Ilse et al., 2018) combina os recortes da foto num unico vetor, que o
classificador transforma na nota da foto; a loss e a entropia cruzada ponderada no
nivel da foto, entao nenhum recorte recebe a nota da foto como se fosse sua.

O mesmo classificador aplicado ao vetor de cada recorte isolado fornece a estimativa
por regiao, e os pesos de atencao mostram quanto cada regiao pesou na decisao.

Protocolo igual ao das outras ablacoes: folds agrupados de results/kfold_splits_v3.json,
Adam lr 1e-4, weight decay 1e-4, ReduceLROnPlateau, early stopping pela acuracia de
validacao (por foto), 40 epocas, patience 10. Sem mixup (as fotos tem numero diferente
de recortes).

Uso:
    cd training
    python run_region_mil_kfold.py --name mil_with --fold 0 --pretrained-backbone models/backbone_scin_pretrained_v2.pth
    python run_region_mil_kfold.py --name mil_without --fold 0 --no-embed ...
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from run_ablation import BACKBONE_OUT_FEATURES, _load_backbone, build_optimizer, build_transforms, get_device, set_seed
from run_kfold_clean import index_crops


class RegionMILNet(nn.Module):
    def __init__(self, use_embed=True, pretrained_backbone_path=None, num_classes=4, num_regions=5):
        super().__init__()
        self.backbone = _load_backbone(pretrained_backbone_path, arch="resnet18")
        feat_dim = BACKBONE_OUT_FEATURES["resnet18"]
        self.use_embed = use_embed
        dim = feat_dim + (64 if use_embed else 0)
        if use_embed:
            self.region_embed = nn.Embedding(num_regions, 64)
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, imgs, regions, photo_idx, num_photos, with_crops=False):
        h = self.backbone(imgs).flatten(1)
        if self.use_embed:
            h = torch.cat([h, self.region_embed(regions.long())], dim=1)
        s = self.attention(h).squeeze(1)
        # softmax da atencao dentro de cada foto
        s_max = torch.full((num_photos,), -1e9, device=s.device).scatter_reduce(0, photo_idx, s, reduce="amax")
        e = torch.exp(s - s_max[photo_idx])
        denom = torch.zeros(num_photos, device=s.device).index_add(0, photo_idx, e)
        a = e / denom[photo_idx]
        pooled = torch.zeros(num_photos, h.size(1), device=h.device).index_add(0, photo_idx, a.unsqueeze(1) * h)
        logits = self.classifier(pooled)
        if with_crops:
            return logits, self.classifier(h), a
        return logits


class PhotoDataset(Dataset):
    def __init__(self, photos, transform):
        self.photos = photos  # lista de (name, label, [(path, region), ...])
        self.transform = transform

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        name, label, crops = self.photos[idx]
        imgs = np.stack([self.transform(Image.open(p).convert("RGB")).numpy() for p, _ in crops])
        regions = np.array([r for _, r in crops], dtype=np.int64)
        return imgs, regions, np.int64(label), np.int64(idx)


def photo_collate(batch):
    """Montado em numpy no worker; tensores so no processo principal (ver run_region_count_kfold.py)."""
    imgs = np.concatenate([b[0] for b in batch])
    regions = np.concatenate([b[1] for b in batch])
    photo_idx = np.concatenate([np.full(len(b[1]), i, dtype=np.int64) for i, b in enumerate(batch)])
    labels = np.array([b[2] for b in batch])
    ids = np.array([b[3] for b in batch])
    return imgs, regions, photo_idx, labels, ids


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
                                   num_workers=workers, collate_fn=photo_collate, persistent_workers=workers > 0))


def build_photos(index, names):
    photos = []
    for n in names:
        crops = index.get(n, [])
        if not crops:
            continue
        labels = {c[2] for c in crops}
        assert len(labels) == 1, n
        crops = sorted(crops, key=lambda c: c[3])
        photos.append((n, crops[0][2], [(c[1], c[3]) for c in crops]))
    return photos


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for imgs, regions, photo_idx, labels, _ in loader:
        imgs, regions, photo_idx, labels = imgs.to(device), regions.to(device), photo_idx.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs, regions, photo_idx, labels.size(0)), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, crop_probs, crop_regions, crop_photo, attn = [], [], [], [], []
    for imgs, regions, photo_idx, labels, ids in loader:
        logits, crop_logits, a = model(imgs.to(device), regions.to(device), photo_idx.to(device), labels.size(0),
                                       with_crops=True)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        crop_probs.append(F.softmax(crop_logits, dim=1).cpu().numpy())
        crop_regions.append(regions.numpy())
        crop_photo.append(ids.numpy()[photo_idx.numpy()])
        attn.append(a.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(crop_probs), np.concatenate(crop_regions),
            np.concatenate(crop_photo), np.concatenate(attn))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--splits", default="results/kfold_splits_v3.json")
    p.add_argument("--data-dir", default="../data/final_v3")
    p.add_argument("--output-dir", default="results/kfold")
    p.add_argument("--pretrained-backbone", default=None)
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8, help="fotos por lote (~27 recortes)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-photos", type=int, default=0, help="so para teste rapido")
    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.name}] fold {args.fold} ja concluido, pulando")
        return

    fold = json.load(open(args.splits))["folds"][args.fold]
    index = index_crops(args.data_dir)
    train_p, val_p, test_p = (build_photos(index, fold[k]) for k in ("train", "val", "test"))
    if args.max_photos:
        rng = np.random.default_rng(0)
        train_p, val_p, test_p = ([ps[i] for i in rng.permutation(len(ps))[:args.max_photos]]
                                  for ps in (train_p, val_p, test_p))
    n_crops = lambda ps: sum(len(c) for _, _, c in ps)
    print(f"[{args.name}] fold {args.fold}: fotos train={len(train_p)} val={len(val_p)} test={len(test_p)}; "
          f"recortes {n_crops(train_p)}/{n_crops(val_p)}/{n_crops(test_p)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=False, size=224)
    train_loader = make_loader(PhotoDataset(train_p, train_tf), args.batch_size, True, args.workers, drop_last=True)
    val_loader = make_loader(PhotoDataset(val_p, val_tf), args.batch_size, False, args.workers)
    test_loader = make_loader(PhotoDataset(test_p, val_tf), args.batch_size, False, args.workers)

    y_train = [ph[1] for ph in train_p]
    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train), dtype=torch.float)

    set_seed(args.seed)
    model = RegionMILNet(use_embed=not args.no_embed, pretrained_backbone_path=args.pretrained_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(model, "adam", args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    start = time.time()
    best_val, best_state, wait = -1.0, None, 0
    y_val = [ph[1] for ph in val_p]
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device)
        vp = predict(model, val_loader, device)[0]
        val_acc = accuracy_score(y_val, vp.argmax(1))
        scheduler.step(val_acc)
        print(f"[{args.name}] fold {args.fold} epoch {epoch}: val_acc={val_acc:.4f} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    probs, crop_probs, crop_regions, crop_photo, attn = predict(model, test_loader, device)
    names = np.array([ph[0] for ph in test_p])
    labels = np.array([ph[1] for ph in test_p])
    pred = probs.argmax(1)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(labels, pred, weights="quadratic")
    minutes = (time.time() - start) / 60
    print(f"[{args.name}] fold {args.fold}: Acc={acc:.4f} F1m={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)
    np.savez(
        npz_path,
        names=names,
        labels=labels,
        probs=probs,
        crop_names=names[crop_photo],
        crop_regions=crop_regions,
        crop_labels=labels[crop_photo],
        crop_probs=crop_probs,
        attention=attn,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
