"""
Reavalia o ensemble+TTA ja treinado, mas quebrando o resultado por regiao
anatomica -- nao treina nada novo, so reusa os checkpoints existentes.

Uso:
    cd training
    python evaluate_by_region.py --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from run_ablation import ResNet18WithEmbed, ResNet18NoEmbed, REGIONS


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tta_transforms(n_views=5):
    base = [transforms.Resize((224, 224))]
    views = [transforms.Compose(base + [transforms.ToTensor()])]
    variations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation((5, 5)),
        transforms.RandomRotation((-5, -5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation((10, 10)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
    for v in variations[: max(0, n_views - 1)]:
        views.append(transforms.Compose(base + [v, transforms.ToTensor()]))
    return views[:n_views]


@torch.no_grad()
def predict_probs(model, loader, device, region_idx):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        regions = torch.full((imgs.size(0),), region_idx, device=device)
        out = model(imgs, regions)
        probs = F.softmax(out, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--checkpoints-dir", default="models/ablation_ckpt_clean_swa")
    parser.add_argument("--tag", default="with_embedding")
    parser.add_argument("--tta-views", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, f"{args.tag}_seed*.pth")))
    print(f"Checkpoints: {len(ckpts)}")
    tta_views = build_tta_transforms(args.tta_views)
    model_cls = ResNet18WithEmbed if args.tag == "with_embedding" else ResNet18NoEmbed

    print(f"{'Regiao':<14}{'Precisao':>10}{'Recall':>10}{'F1':>10}{'Suporte':>10}")
    for region_idx, region in enumerate(REGIONS):
        region_dir = os.path.join(args.data_dir, region, "test")
        if not os.path.isdir(region_dir):
            continue

        all_model_probs = []
        labels_ref = None
        for ckpt_path in ckpts:
            model = model_cls().to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)

            view_probs = []
            for tf in tta_views:
                ds = datasets.ImageFolder(region_dir, transform=tf)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                probs, labels = predict_probs(model, loader, device, region_idx)
                view_probs.append(probs)
                if labels_ref is None:
                    labels_ref = labels
            all_model_probs.append(np.mean(view_probs, axis=0))
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        ensemble_probs = np.mean(all_model_probs, axis=0)
        preds = ensemble_probs.argmax(axis=1)
        p, r, f1, _ = precision_recall_fscore_support(labels_ref, preds, average="weighted", zero_division=0)
        print(f"{region:<14}{p:>10.2f}{r:>10.2f}{f1:>10.2f}{len(labels_ref):>10}")


if __name__ == "__main__":
    main()
