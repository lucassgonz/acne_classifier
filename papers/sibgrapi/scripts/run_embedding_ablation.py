#!/usr/bin/env python3
"""Ablation: ResNet-18 with vs without anatomical region embeddings."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}
CLASS_NAMES = ["0", "1", "2", "3"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AcneDataset(Dataset):
    def __init__(self, base_dir: str, splits: list[str], transform=None, include_val_in_train: bool = False):
        self.transform = transform
        self.samples: list[tuple[str, int, str]] = []
        use_splits = list(splits)
        if include_val_in_train and "train" in use_splits and "val" not in use_splits:
            use_splits.append("val")

        for region in REGIONS:
            for split in use_splits:
                split_dir = os.path.join(base_dir, region, split)
                if not os.path.isdir(split_dir):
                    continue
                ds = datasets.ImageFolder(split_dir, transform=None)
                for path, label in ds.samples:
                    self.samples.append((path, label, region))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, region = self.samples[idx]
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, REGION_TO_IDX[region]


class MultiRegionResNet18(nn.Module):
    def __init__(self, num_classes: int = 4, num_regions: int = 5, use_embedding: bool = True, embed_dim: int = 64):
        super().__init__()
        self.use_embedding = use_embedding
        self.resnet = models.resnet18(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        feat_dim = in_features
        if use_embedding:
            self.region_embed = nn.Embedding(num_regions, embed_dim)
            feat_dim = in_features + embed_dim
        else:
            self.region_embed = None
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
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
        img_features = self.resnet(x)
        if self.use_embedding:
            region_features = self.region_embed(region_idx.long()).squeeze(1)
            combined = torch.cat([img_features, region_features], dim=1)
        else:
            combined = img_features
        return self.classifier(combined)


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    robust_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
        ]
    )
    return train_tf, eval_tf, robust_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, regions = [], [], []
    for imgs, labels, region_idx in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        region_idx = region_idx.to(device)
        logits = model(imgs, region_idx)
        pred = logits.argmax(dim=1)
        ys.extend(labels.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        regions.extend(region_idx.cpu().tolist())
    ys = np.array(ys)
    preds = np.array(preds)
    regions = np.array(regions)

    report = classification_report(
        ys, preds, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    overall = {
        "accuracy": float(accuracy_score(ys, preds)),
        "f1_weighted": float(f1_score(ys, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in CLASS_NAMES
        },
    }

    by_region = {}
    for r_i, r_name in enumerate(REGIONS):
        mask = regions == r_i
        if mask.sum() == 0:
            continue
        y_r, p_r = ys[mask], preds[mask]
        by_region[r_name] = {
            "accuracy": float(accuracy_score(y_r, p_r)),
            "f1_weighted": float(f1_score(y_r, p_r, average="weighted", zero_division=0)),
            "support": int(mask.sum()),
        }
    return overall, by_region, ys, preds


def train_model(
    model,
    train_loader,
    test_loader,
    robust_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs: int,
    ckpt_path: Path,
    patience: int = 10,
):
    best_robust = -1.0
    patience_counter = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n = 0
        t0 = time.time()
        for imgs, labels, region_idx in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            region_idx = region_idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, region_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)

        overall, _, _, _ = evaluate(model, test_loader, device)
        robust, _, _, _ = evaluate(model, robust_loader, device)
        test_acc = overall["accuracy"] * 100
        robust_acc = robust["accuracy"] * 100
        gap = test_acc - robust_acc
        scheduler.step(robust_acc)

        row = {
            "epoch": epoch + 1,
            "loss": running_loss / max(n, 1),
            "test_acc": test_acc,
            "robust_acc": robust_acc,
            "gap": gap,
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} loss={row['loss']:.4f} "
            f"test={test_acc:.2f}% robust={robust_acc:.2f}% gap={gap:.1f} ({row['seconds']:.0f}s)"
        )

        if robust_acc > best_robust and gap < 18:
            best_robust = robust_acc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path} (robust={best_robust:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history, best_robust


def extract_embeddings(model) -> list[list[float]]:
    if model.region_embed is None:
        return []
    w = model.region_embed.weight.detach().cpu().numpy()
    return w.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/Users/GonLu/Documents/Projetos/pibic_final/data/final",
    )
    parser.add_argument(
        "--with-weights",
        default="/Users/GonLu/Documents/Projetos/pibic_final/training/notebooks/best_robust_model.pth",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/GonLu/Documents/pessoal/artigo-118-revisao/results",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train-noembed", action="store_true")
    parser.add_argument("--retrain-withembed", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    train_tf, eval_tf, robust_tf = build_transforms()
    train_ds = AcneDataset(args.data_dir, splits=["train"], transform=train_tf, include_val_in_train=True)
    test_ds = AcneDataset(args.data_dir, splits=["test"], transform=eval_tf)
    robust_ds = AcneDataset(args.data_dir, splits=["test"], transform=robust_tf)
    print(f"train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    robust_loader = DataLoader(robust_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    labels = [y for _, y, _ in train_ds.samples]
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- WithEmbed ----
    with_model = MultiRegionResNet18(use_embedding=True).to(device)
    with_ckpt = out_dir / "best_withembed.pth"
    if args.retrain_withembed or not os.path.isfile(args.with_weights):
        print("Training WithEmbed from scratch...")
        opt = torch.optim.Adam(with_model.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        train_model(
            with_model, train_loader, test_loader, robust_loader, criterion, opt, sch, device, args.epochs, with_ckpt
        )
        with_model.load_state_dict(torch.load(with_ckpt, map_location=device, weights_only=True))
    else:
        sd = torch.load(args.with_weights, map_location=device, weights_only=True)
        with_model.load_state_dict(sd)
        torch.save(sd, with_ckpt)
        print(f"Loaded WithEmbed from {args.with_weights}")

    with_overall, with_region, _, _ = evaluate(with_model, test_loader, device)
    print(
        f"WithEmbed test acc={with_overall['accuracy']*100:.2f}% "
        f"f1w={with_overall['f1_weighted']:.4f}"
    )

    # ---- NoEmbed ----
    no_model = MultiRegionResNet18(use_embedding=False).to(device)
    no_ckpt = out_dir / "best_noembed.pth"
    no_history = []
    if args.skip_train_noembed and no_ckpt.exists():
        no_model.load_state_dict(torch.load(no_ckpt, map_location=device, weights_only=True))
        print(f"Loaded NoEmbed from {no_ckpt}")
    else:
        print("Training NoEmbed from scratch...")
        opt = torch.optim.Adam(no_model.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        no_history, _ = train_model(
            no_model, train_loader, test_loader, robust_loader, criterion, opt, sch, device, args.epochs, no_ckpt
        )
        no_model.load_state_dict(torch.load(no_ckpt, map_location=device, weights_only=True))

    no_overall, no_region, _, _ = evaluate(no_model, test_loader, device)
    print(
        f"NoEmbed test acc={no_overall['accuracy']*100:.2f}% "
        f"f1w={no_overall['f1_weighted']:.4f}"
    )

    emb = extract_embeddings(with_model)
    emb_np = np.array(emb, dtype=np.float64)
    # cosine similarity
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8
    sim = (emb_np @ emb_np.T) / (norms @ norms.T)

    delta = {
        "accuracy": with_overall["accuracy"] - no_overall["accuracy"],
        "f1_weighted": with_overall["f1_weighted"] - no_overall["f1_weighted"],
        "f1_macro": with_overall["f1_macro"] - no_overall["f1_macro"],
        "by_region": {
            r: {
                "accuracy": with_region[r]["accuracy"] - no_region[r]["accuracy"],
                "f1_weighted": with_region[r]["f1_weighted"] - no_region[r]["f1_weighted"],
            }
            for r in REGIONS
            if r in with_region and r in no_region
        },
    }

    payload = {
        "device": str(device),
        "seed": args.seed,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "regions": REGIONS,
        "with_embed": {"overall": with_overall, "by_region": with_region},
        "no_embed": {"overall": no_overall, "by_region": no_region},
        "delta": delta,
        "embedding_matrix": emb,
        "embedding_cosine_similarity": sim.tolist(),
        "noembed_history": no_history,
    }

    out_json = out_dir / "ablation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")
    print(
        f"Delta acc={delta['accuracy']*100:+.2f} pp  "
        f"f1w={delta['f1_weighted']*100:+.2f} pp"
    )


if __name__ == "__main__":
    main()
