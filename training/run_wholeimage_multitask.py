"""
Treina o backbone CBAM (Section IV-E/F do artigo) com aprendizado CONJUNTO
severidade+contagem de licoes, replicando a supervisao usada pelo JAGC
(Wu et al., ICCV2019) -- mas com a contagem REAL e oficial do proprio
release do ACNE04 (baixada em utils/build_official_acne04_split.py /
add_lesion_counts_to_official_split.py), nao uma estimativa.

So roda no split OFICIAL (results/kfold_splits_official_acne04.json),
que e onde temos contagem real pra TODAS as imagens (AcneLevel = release
oficial, 1457 imagens) -- objetivo direto: fechar o gap de 7,47pp contra
o JAGC (76,64% vs 84,11%) identificado na Secao IV-F do artigo.

Script isolado (nao reusa train_model() de run_ablation.py) pra nao
arriscar quebrar nada do pipeline ja validado -- cabeca de regressao
extra, loss combinada CE + lambda*SmoothL1(log1p(contagem)).

Uso:
    cd training
    python run_wholeimage_multitask.py --name official_multitask --fold 0 \
        --pretrained-backbone models/backbone_scin_pretrained_v2.pth --mixup-alpha 0.2 \
        --cbam --swa --swa-epochs 10 --count-loss-weight 0.2
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from run_ablation import CBAM, _load_backbone_features, BACKBONE_OUT_FEATURES, build_transforms, get_device, set_seed


class ResNet18CBAMMultiTask(nn.Module):
    """Mesmo backbone+CBAM do ResNet18CBAM, mas com DUAS cabecas sobre a
    mesma representacao agrupada: classificacao (4 graus) e regressao
    (log1p da contagem de licoes) -- supervisao auxiliar replicando o
    JAGC, usando contagem REAL (nao estimada)."""

    def __init__(self, num_classes=4, pretrained_backbone_path=None, arch="resnet18"):
        super().__init__()
        self.backbone = _load_backbone_features(pretrained_backbone_path, arch=arch)
        in_features = BACKBONE_OUT_FEATURES[arch]
        self.cbam = CBAM(in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.count_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        shared = self.trunk(feat)
        return self.cls_head(shared), self.count_head(shared).squeeze(-1)


class WholeImageCountDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (name, path, label, log_count)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, label, log_count = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, torch.tensor(log_count, dtype=torch.float32)


def build_tta_transforms(n_views, size=224):
    base = [T.Resize((size, size))]
    views = [T.Compose(base + [T.ToTensor()])]
    variations = [
        T.RandomHorizontalFlip(p=1.0),
        T.RandomRotation((5, 5)),
        T.RandomRotation((-5, -5)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomRotation((10, 10)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
    for v in variations[: max(0, n_views - 1)]:
        views.append(T.Compose(base + [v, T.ToTensor()]))
    return views[:n_views]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, _ in loader:
        logits, _ = model(imgs.to(device))
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return acc, f1, qwk, probs


@torch.no_grad()
def predict_tta(model, samples, tta_views, device, batch_size):
    model.eval()
    view_probs = []
    for tf in tta_views:
        loader = DataLoader(WholeImageCountDataset(samples, tf), batch_size=batch_size, shuffle=False, num_workers=0)
        probs = []
        for imgs, _, _ in loader:
            logits, _ = model(imgs.to(device))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        view_probs.append(np.concatenate(probs))
    return np.mean(view_probs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--splits", default="results/kfold_splits_official_acne04.json")
    p.add_argument("--dataset-root", default="../Dataset")
    p.add_argument("--output-dir", default="results/kfold_wholeimage")
    p.add_argument("--pretrained-backbone", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--swa", action="store_true")
    p.add_argument("--swa-epochs", type=int, default=10)
    p.add_argument("--tta-views", type=int, default=1)
    p.add_argument("--count-loss-weight", type=float, default=0.2,
                    help="Peso da loss de regressao de contagem relativa a classificacao (CE).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.name}] fold {args.fold} ja concluido, pulando")
        return

    with open(args.splits) as f:
        fold = json.load(f)["folds"][args.fold]

    file_index = {}
    for pth in glob.glob(os.path.join(args.dataset_root, "Train", "Level *", "*.jpg")) + \
                glob.glob(os.path.join(args.dataset_root, "Validation", "Level *", "*.jpg")):
        file_index[os.path.basename(pth)] = pth

    def gather(split_key):
        labels = fold[f"{split_key}_labels_official"]
        counts = fold[f"{split_key}_lesion_counts"]
        out = []
        for n, label in labels.items():
            if n in file_index:
                out.append((n, file_index[n], label, float(np.log1p(counts[n]))))
        return out

    train_s, val_s, test_s = (gather(k) for k in ("train", "val", "test"))
    print(f"[{args.name}] fold {args.fold}: images train={len(train_s)} val={len(val_s)} test={len(test_s)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=False, size=args.img_size)
    train_loader = DataLoader(WholeImageCountDataset(train_s, train_tf), batch_size=args.batch_size,
                               shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(WholeImageCountDataset(val_s, val_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(WholeImageCountDataset(test_s, val_tf), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    y_train = [s[2] for s in train_s]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train), dtype=torch.float
    ).to(device)

    set_seed(args.seed)
    model = ResNet18CBAMMultiTask(pretrained_backbone_path=args.pretrained_backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    count_loss_fn = nn.SmoothL1Loss()

    def mixup(x, y, alpha):
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    best_qwk, best_state, patience_ct = -1, None, 0
    start = time.time()
    swa_model = AveragedModel(model) if args.swa else None

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels, log_counts in train_loader:
            imgs, labels, log_counts = imgs.to(device), labels.to(device), log_counts.to(device)
            imgs_mixed, ya, yb, lam = mixup(imgs, labels, args.mixup_alpha)
            optimizer.zero_grad()
            logits, pred_log_count = model(imgs_mixed)
            ce = lam * F.cross_entropy(logits, ya, weight=class_weights) + \
                 (1 - lam) * F.cross_entropy(logits, yb, weight=class_weights)
            count_loss = count_loss_fn(pred_log_count, log_counts)
            loss = ce + args.count_loss_weight * count_loss
            loss.backward()
            optimizer.step()

        acc, f1, qwk, _ = evaluate(model, val_loader, device)
        scheduler.step(qwk)
        if qwk > best_qwk:
            best_qwk, best_state, patience_ct = qwk, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= args.patience:
                print(f"  early stop na epoca {epoch}", flush=True)
                break

    model.load_state_dict(best_state)

    if args.swa:
        swa_model = AveragedModel(model)
        swa_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 0.1)
        for _ in range(args.swa_epochs):
            model.train()
            for imgs, labels, log_counts in train_loader:
                imgs, labels, log_counts = imgs.to(device), labels.to(device), log_counts.to(device)
                swa_optimizer.zero_grad()
                logits, pred_log_count = model(imgs)
                loss = F.cross_entropy(logits, labels, weight=class_weights) + \
                       args.count_loss_weight * count_loss_fn(pred_log_count, log_counts)
                loss.backward()
                swa_optimizer.step()
            swa_model.update_parameters(model)
        update_bn(train_loader, swa_model, device=device)
        model = swa_model.module

    if args.tta_views > 1:
        tta_transforms = build_tta_transforms(args.tta_views, size=args.img_size)
        probs = predict_tta(model, test_s, tta_transforms, device, args.batch_size)
    else:
        _, _, _, probs = evaluate(model, test_loader, device)

    labels = np.array([s[2] for s in test_s])
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    minutes = (time.time() - start) / 60
    print(f"[{args.name}] fold {args.fold}: Acc={acc:.4f} F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)

    np.savez(
        npz_path,
        names=np.array([s[0] for s in test_s]),
        labels=labels,
        probs=probs,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
