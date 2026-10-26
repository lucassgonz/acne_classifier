"""
Replica fielmente a TECNICA do JAGC (Wu et al., ICCV2019,
github.com/xpwu95/LDL/blob/master/code/train.py e
code/model/resnet50.py, lidos diretamente desta sessao) -- mas com o
backbone ResNet-18+CBAM proprio deste trabalho (nao ResNet-50), mantendo
a "essencia" do artigo (arquitetura leve, region-aware por design,
pretrain SCIN) e so trocando a SUPERVISAO pela deles.

Diferenca pro run_wholeimage_multitask.py (regressao escalar simples,
ja testado, +0,41pp/+1,01pp macroF1): aqui a contagem de licoes e
modelada como uma DISTRIBUICAO gaussiana sobre 65 valores possiveis
(0-64), nao um numero so; o rotulo de classificacao tambem vira uma
distribuicao suave derivada dessa gaussiana (nao mais rotulo rigido);
ha uma terceira saida (cou2cls, soma deterministica dos "baldes" de
contagem que correspondem a cada grau) com uma loss de consistencia
extra ligando contagem e classificacao. 3 losses KL-divergence
combinadas com os mesmos pesos do paper original (sigma=3.0, lam=0.6).

Uso:
    cd training
    python run_wholeimage_jagc.py --name official_jagc --fold 0 \
        --pretrained-backbone models/backbone_scin_pretrained_v2.pth --mixup-alpha 0.2 \
        --swa --swa-epochs 10 --tta-views 7
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

NUM_COUNT_CLASSES = 65
SIGMA = 3.0   # desvio da gaussiana sobre contagem -- igual ao paper original (30*0.1)
LAM = 0.6     # peso classificacao vs contagem na loss combinada -- igual ao paper original (6*0.1)
BUCKETS = [(0, 5), (5, 20), (20, 50), (50, NUM_COUNT_CLASSES)]  # mesmos limites do paper (criterion())


def gen_label_distribution(counts, sigma=SIGMA, num_count_classes=NUM_COUNT_CLASSES):
    """Gaussiana sobre o espaco de contagem (0..64), centrada na contagem
    real -- replica utils/genLD.py do repo original (caso 'klloss')."""
    counts = np.clip(np.asarray(counts, dtype=np.float64), 0, num_count_classes - 1)
    label_set = np.arange(num_count_classes).reshape(-1, 1)
    dif = label_set - counts.reshape(1, -1)
    ld = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(dif ** 2) / (2 * sigma ** 2))
    ld = ld / ld.sum(axis=0, keepdims=True)
    return ld.T  # (N, 65)


def bucket_sum(dist_65):
    """Soma a distribuicao de 65 vias nos 4 'baldes' de gravidade --
    replica a linha cou2cls = stack(sum(cou[:, :5]), ...) do forward()."""
    return np.stack([dist_65[:, lo:hi].sum(axis=1) for lo, hi in BUCKETS], axis=1)


class ResNet18CBAMJAGC(nn.Module):
    """Backbone ResNet-18+CBAM (nosso) com DUAS cabecas (classificacao e
    contagem-distribuicao) sobre a mesma representacao compartilhada --
    equivalente estrutural ao `fc`/`counting` do ResNet(...) original,
    so que com um tronco MLP (nao Linear puro) pra manter a capacidade
    do classificador que ja usamos nos demais experimentos."""

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
            nn.Dropout(0.2), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, num_classes),
        )
        self.count_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, NUM_COUNT_CLASSES),
        )

    def forward(self, x):
        feat = self.pool(self.cbam(self.backbone(x))).flatten(1)
        shared = self.trunk(feat)
        cls = F.softmax(self.cls_head(shared), dim=1) + 1e-4
        cou = F.softmax(self.count_head(shared), dim=1) + 1e-4
        cou2cls = torch.stack([cou[:, lo:hi].sum(dim=1) for lo, hi in BUCKETS], dim=1)
        return cls, cou, cou2cls


class WholeImageCountDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # (name, path, label, count)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, label, count = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, count


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


def jagc_loss(cls, cou, cou2cls, counts_np, device):
    """As 3 KL losses + combinacao, exatamente como no train.py original."""
    ld = gen_label_distribution(np.asarray(counts_np) - 1.0)  # contagem 1..65 -> indice 0..64 (b_l - 1 no original)
    ld_4 = bucket_sum(ld)
    ld_t = torch.from_numpy(ld).float().to(device)
    ld4_t = torch.from_numpy(ld_4).float().to(device)

    kl = nn.KLDivLoss(reduction="mean")
    loss_cls = kl(torch.log(cls), ld4_t) * 4.0
    loss_cou = kl(torch.log(cou), ld_t) * NUM_COUNT_CLASSES
    loss_cls_cou = kl(torch.log(cou2cls), ld4_t) * 4.0
    loss = (loss_cls + loss_cls_cou) * 0.5 * LAM + loss_cou * (1.0 - LAM)
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_cls, all_cou2cls, all_labels = [], [], []
    for imgs, labels, _ in loader:
        cls, _, cou2cls = model(imgs.to(device))
        all_cls.append(cls.cpu().numpy())
        all_cou2cls.append(cou2cls.cpu().numpy())
        all_labels.append(labels.numpy())
    cls = np.concatenate(all_cls)
    cou2cls = np.concatenate(all_cou2cls)
    labels = np.concatenate(all_labels)
    combo = cls + cou2cls
    preds = combo.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return acc, f1, qwk, cls, cou2cls


@torch.no_grad()
def predict_tta(model, samples, tta_views, device, batch_size):
    model.eval()
    view_cls, view_cou2cls = [], []
    for tf in tta_views:
        loader = DataLoader(WholeImageCountDataset(samples, tf), batch_size=batch_size, shuffle=False, num_workers=0)
        cls_probs, cou2cls_probs = [], []
        for imgs, _, _ in loader:
            cls, _, cou2cls = model(imgs.to(device))
            cls_probs.append(cls.cpu().numpy())
            cou2cls_probs.append(cou2cls.cpu().numpy())
        view_cls.append(np.concatenate(cls_probs))
        view_cou2cls.append(np.concatenate(cou2cls_probs))
    return np.mean(view_cls, axis=0), np.mean(view_cou2cls, axis=0)


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest-v2", default=None,
                    help="Usa Acne04+AcneLevel(tratado) (ex: ../data/final_wholeimage_v2/manifest.json) com --splits "
                         "results/kfold_splits_wholeimage_v2.json; a contagem de licoes vem do split oficial via identidade canonica.")
    p.add_argument("--count-source", default="results/kfold_splits_official_acne04.json")
    p.add_argument("--fixed-schedule", action="store_true",
                    help="Usa o cronograma de treino do JAGC original (train.py): SGD+momentum, "
                         "LR em escada (--lr-milestones), treina todas as --epochs sem early stop. "
                         "O resto (backbone pretreinado, CBAM, mixup, SWA, TTA) continua igual.")
    p.add_argument("--lr-milestones", default="30,60,90",
                    help="Epocas (separadas por virgula) onde o LR e multiplicado por --lr-decay, so com --fixed-schedule.")
    p.add_argument("--lr-decay", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.name}] fold {args.fold} ja concluido, pulando")
        return

    with open(args.splits) as f:
        fold = json.load(f)["folds"][args.fold]

    if args.manifest_v2:
        manifest = {m["name"]: m for m in json.load(open(args.manifest_v2))}
        counts_by_canon = {}
        for fd in json.load(open(args.count_source))["folds"]:
            for sk in ("train", "val", "test"):
                counts_by_canon.update(fd[f"{sk}_lesion_counts"])

        def gather(split_key):
            return [(n, manifest[n]["path"], manifest[n]["level"], float(counts_by_canon[manifest[n]["canonical"]]))
                    for n in fold[split_key] if n in manifest]
    else:
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
                    out.append((n, file_index[n], label, float(counts[n])))
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

    set_seed(args.seed)
    model = ResNet18CBAMJAGC(pretrained_backbone_path=args.pretrained_backbone, arch=args.arch).to(device)
    if args.fixed_schedule:
        lr_milestones = [int(x) for x in args.lr_milestones.split(",")]
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    def mixup(x, counts, alpha):
        if alpha <= 0:
            return x, counts
        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(counts))
        mixed_x = lam * x + (1 - lam) * x[idx]
        mixed_counts = lam * counts + (1 - lam) * counts[idx]
        return mixed_x, mixed_counts

    best_qwk, best_state, patience_ct = -1, None, 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels, counts in train_loader:
            imgs = imgs.to(device)
            counts_np = counts.numpy()
            imgs_mixed, counts_mixed = mixup(imgs, counts_np, args.mixup_alpha)
            optimizer.zero_grad()
            cls, cou, cou2cls = model(imgs_mixed)
            loss = jagc_loss(cls, cou, cou2cls, counts_mixed, device)
            loss.backward()
            optimizer.step()

        acc, f1, qwk, _, _ = evaluate(model, val_loader, device)
        if args.fixed_schedule:
            scheduler.step()
        else:
            scheduler.step(qwk)
        if qwk > best_qwk:
            best_qwk, best_state, patience_ct = qwk, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if not args.fixed_schedule and patience_ct >= args.patience:
                print(f"  early stop na epoca {epoch}", flush=True)
                break

    model.load_state_dict(best_state)

    if args.swa:
        swa_model = AveragedModel(model)
        swa_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 0.1)
        for _ in range(args.swa_epochs):
            model.train()
            for imgs, labels, counts in train_loader:
                imgs = imgs.to(device)
                counts_np = counts.numpy()
                swa_optimizer.zero_grad()
                cls, cou, cou2cls = model(imgs)
                loss = jagc_loss(cls, cou, cou2cls, counts_np, device)
                loss.backward()
                swa_optimizer.step()
            swa_model.update_parameters(model)
        update_bn(train_loader, swa_model, device=device)
        model = swa_model.module

    if args.tta_views > 1:
        tta_transforms = build_tta_transforms(args.tta_views, size=args.img_size)
        cls_probs, cou2cls_probs = predict_tta(model, test_s, tta_transforms, device, args.batch_size)
    else:
        _, _, _, cls_probs, cou2cls_probs = evaluate(model, test_loader, device)

    combo_probs = cls_probs + cou2cls_probs
    labels = np.array([s[2] for s in test_s])
    preds = combo_probs.argmax(axis=1)
    preds_cls_only = cls_probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    acc_cls_only = accuracy_score(labels, preds_cls_only)
    f1 = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    minutes = (time.time() - start) / 60
    print(f"[{args.name}] fold {args.fold}: Acc(cls+cou2cls)={acc:.4f} Acc(cls only)={acc_cls_only:.4f} "
          f"F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)

    np.savez(
        npz_path,
        names=np.array([s[0] for s in test_s]),
        labels=labels,
        probs=combo_probs / combo_probs.sum(axis=1, keepdims=True),
        probs_cls_only=cls_probs,
        metrics=np.array([acc, f1, qwk]),
        minutes=np.array([minutes]),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
