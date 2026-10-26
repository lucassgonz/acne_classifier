"""
Validacao cruzada k-fold (k=5) do baseline "imagem inteira" (sem recorte por
regiao), usando os MESMOS folds por grupo (livres de quase-duplicata entre
treino/teste) de utils/build_kfold_splits.py que o pipeline region-aware usa
-- para que a comparacao entre as duas abordagens seja sobre o split limpo,
nao o split antigo potencialmente vazado usado em data/wholeimage/.

Cada execucao treina UM fold e salva as probabilidades de teste por imagem,
permitindo agregar as 5 folds e comparar configuracoes (ex: loss ordinal vs
cross-entropy) imagem a imagem.

Uso:
    cd training
    python run_wholeimage_kfold.py --name whole_base --fold 0 \
        --pretrained-backbone models/backbone_scin_pretrained_v2.pth --mixup-alpha 0.2
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from run_ablation import ResNet18NoEmbed, ResNet18CBAM, build_transforms, get_device, set_seed, train_model


def build_tta_transforms(n_views, size=224):
    """Mesma receita determinística de evaluate_wholeimage_ensemble_tta.py:
    identidade, espelho horizontal, pequenas rotações e color jitter."""
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

WHOLE_IMAGE_ROOTS = [
    ("../Dataset/Train/Level 0", 0), ("../Dataset/Train/Level 1", 1),
    ("../Dataset/Train/Level 2", 2), ("../Dataset/Train/Level 3", 3),
    ("../Dataset/Validation/Level 0", 0), ("../Dataset/Validation/Level 1", 1),
    ("../Dataset/Validation/Level 2", 2), ("../Dataset/Validation/Level 3", 3),
]
MANUAL_ROOTS = [
    ("../data/manual_annotations/0", 0), ("../data/manual_annotations/1", 1),
    ("../data/manual_annotations/2", 2), ("../data/manual_annotations/3", 3),
]


def index_whole_images():
    index = {}
    for root, label in WHOLE_IMAGE_ROOTS:
        for p in glob.glob(os.path.join(root, "*.jpg")):
            index[os.path.basename(p)] = (p, label)
    for root, label in MANUAL_ROOTS:
        for p in glob.glob(os.path.join(root, "*.jpg")):
            index["manual_" + os.path.basename(p)] = (p, label)
    return index


def index_whole_images_v2(manifest_path):
    """Acne04 + AcneLevel(tratado), sem manual_annotations (ver
    utils/build_wholeimage_manifest_v2.py)."""
    manifest = json.load(open(manifest_path))
    return {m["name"]: (m["path"], m["level"]) for m in manifest}


class WholeImageSplitDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (name, path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, torch.tensor(-1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for imgs, _, _ in loader:
        probs.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def predict_tta(model, test_s, tta_views, device, batch_size):
    """Media as probabilidades softmax de varias views aumentadas do mesmo
    conjunto de teste (ordem fixa -- 'names'/'labels' continuam alinhados)."""
    model.eval()
    view_probs = []
    for tf in tta_views:
        loader = DataLoader(WholeImageSplitDataset(test_s, tf), batch_size=batch_size,
                             shuffle=False, num_workers=0)
        probs = []
        for imgs, _, _ in loader:
            probs.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
        view_probs.append(np.concatenate(probs))
    return np.mean(view_probs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--splits", default="results/kfold_splits.json")
    p.add_argument("--output-dir", default="results/kfold_wholeimage")
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
    p.add_argument("--tta-views", type=int, default=1)
    p.add_argument("--cbam", action="store_true", help="Insere atencao CBAM entre o backbone e o pooling global.")
    p.add_argument("--acne04-only", action="store_true", help="Usa so as imagens do ACNE04 (exclui as anotadas manualmente).")
    p.add_argument("--manifest-v2", default=None,
                    help="Se informado (ex: ../data/final_wholeimage_v2/manifest.json), usa Acne04+AcneLevel(tratado) "
                         "via utils/build_wholeimage_manifest_v2.py em vez das pastas Dataset/+manual_annotations.")
    p.add_argument("--official-split", action="store_true",
                    help="Usa results/kfold_splits_official_acne04.json (split oficial NNEW_trainval/test do "
                         "JAGC/ICCV2019, ver utils/build_official_acne04_split.py) -- rotulos oficiais embutidos "
                         "no proprio split, nao os da pasta Dataset/Level X. --splits deve apontar pra esse arquivo.")
    p.add_argument("--official-dataset-root", default="../Dataset")
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

    if args.official_split:
        file_index = {}
        for p in glob.glob(os.path.join(args.official_dataset_root, "Train", "Level *", "*.jpg")) + \
                 glob.glob(os.path.join(args.official_dataset_root, "Validation", "Level *", "*.jpg")):
            file_index[os.path.basename(p)] = p

        def gather_official(split_key):
            labels = fold[f"{split_key}_labels_official"]
            out = []
            for n, label in labels.items():
                if n in file_index:
                    out.append((n, file_index[n], label))
            return out

        train_s, val_s, test_s = (gather_official(k) for k in ("train", "val", "test"))
    else:
        index = index_whole_images_v2(args.manifest_v2) if args.manifest_v2 else index_whole_images()

        def gather(names):
            out = []
            for n in names:
                if args.acne04_only and n.startswith("manual_"):
                    continue
                if n in index:
                    path, label = index[n]
                    out.append((n, path, label))
            return out

        train_s, val_s, test_s = (gather(fold[k]) for k in ("train", "val", "test"))
    print(f"[{args.name}] fold {args.fold}: images train={len(train_s)} val={len(val_s)} test={len(test_s)}", flush=True)

    device = get_device()
    train_tf, val_tf = build_transforms(strong_aug=args.strong_aug, size=args.img_size)
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
    model_cls = ResNet18CBAM if args.cbam else ResNet18NoEmbed
    model = model_cls(pretrained_backbone_path=args.pretrained_backbone, arch=args.arch).to(device)

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
    print(f"[{args.name}] fold {args.fold}: Acc={acc:.4f} F1={f1:.4f} QWK={qwk:.4f} ({minutes:.1f} min)", flush=True)

    if args.tta_views > 1:
        tta_views = build_tta_transforms(args.tta_views, size=args.img_size)
        probs = predict_tta(model, test_s, tta_views, device, args.batch_size)
    else:
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
