"""
Ensemble HIBRIDO: combina os modelos nao-ordinais (softmax, treinados com
Mixup) com os modelos CORAL (ordinais) via soft voting -- media das
probabilidades por classe de ambas as familias de modelo, com TTA em cada
uma. Inspirado na abordagem do OviT-LTA (Telkom University), que obteve
seu melhor resultado combinando 3 modelos diferentes por soft voting.

As probabilidades por classe do CORAL sao derivadas das probabilidades
cumulativas (sigmoid dos limiares): P(y=0)=1-P(y>0), P(y=k)=P(y>k-1)-P(y>k).

Uso:
    cd training
    python evaluate_hybrid_ensemble.py \
        --data-dir ../data/final \
        --softmax-ckpt-dir models/ablation_ckpt_mixup \
        --coral-ckpt-dir models/ablation_ckpt_coral \
        --tta-views 5
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

from run_ablation import AcneDataset, ResNet18WithEmbed, CLASS_NAMES
from run_ordinal import CoralResNet18WithEmbed


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_tta_transforms(n_views):
    base = [transforms.Resize((224, 224))]
    views = [transforms.Compose(base + [transforms.ToTensor()])]
    variations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation((5, 5)),
        transforms.RandomRotation((-5, -5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    for v in variations[: max(0, n_views - 1)]:
        views.append(transforms.Compose(base + [v, transforms.ToTensor()]))
    return views[:n_views]


def coral_cumulative_to_class_probs(cum_probs):
    """[N, K-1] P(y>k) -> [N, K] P(y=k), com clip para evitar negativos por ruido numerico."""
    n = cum_probs.shape[0]
    k_minus_1 = cum_probs.shape[1]
    class_probs = np.zeros((n, k_minus_1 + 1))
    class_probs[:, 0] = 1 - cum_probs[:, 0]
    for k in range(1, k_minus_1):
        class_probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    class_probs[:, k_minus_1] = cum_probs[:, k_minus_1 - 1]
    return np.clip(class_probs, 0, None)


@torch.no_grad()
def predict_softmax_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, regions = imgs.to(device), regions.to(device)
        probs = F.softmax(model(imgs, regions), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


@torch.no_grad()
def predict_coral_class_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, regions = imgs.to(device), regions.to(device)
        cum_probs = torch.sigmoid(model(imgs, regions)).cpu().numpy()
        all_probs.append(coral_cumulative_to_class_probs(cum_probs))
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def evaluate_family(ckpts, model_ctor, predict_fn, data_dir, tta_views, device, batch_size):
    """Roda TTA para cada checkpoint de uma familia de modelo, retorna lista de [N,K] probs por modelo."""
    all_probs = []
    labels_ref = None
    for ckpt_path in ckpts:
        model = model_ctor().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        view_probs = []
        for tf in tta_views:
            ds = AcneDataset(data_dir, ["test"], tf)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            probs, labels = predict_fn(model, loader, device)
            view_probs.append(probs)
            if labels_ref is None:
                labels_ref = labels
        all_probs.append(np.mean(view_probs, axis=0))

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"  {os.path.basename(ckpt_path)} ok")
    return all_probs, labels_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--softmax-ckpt-dir", default="models/ablation_ckpt_mixup")
    parser.add_argument("--coral-ckpt-dir", default="models/ablation_ckpt_coral")
    parser.add_argument("--tta-views", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    softmax_ckpts = sorted(glob.glob(os.path.join(args.softmax_ckpt_dir, "with_embedding_seed*.pth")))
    coral_ckpts = sorted(glob.glob(os.path.join(args.coral_ckpt_dir, "seed*.pth")))
    print(f"Modelos softmax (mixup): {len(softmax_ckpts)}")
    print(f"Modelos CORAL (ordinal): {len(coral_ckpts)}")

    tta_views = build_tta_transforms(args.tta_views)

    print("\nAvaliando familia softmax (mixup)...")
    softmax_probs, labels_ref = evaluate_family(
        softmax_ckpts, ResNet18WithEmbed, predict_softmax_probs,
        args.data_dir, tta_views, device, args.batch_size,
    )

    print("\nAvaliando familia CORAL (ordinal)...")
    coral_probs, labels_ref2 = evaluate_family(
        coral_ckpts, CoralResNet18WithEmbed, predict_coral_class_probs,
        args.data_dir, tta_views, device, args.batch_size,
    )
    assert (labels_ref == labels_ref2).all(), "Labels de teste divergem entre as duas familias"

    # Soft voting: media de TODAS as probabilidades (softmax + CORAL convertido), todas em [N,K]
    all_probs = softmax_probs + coral_probs
    hybrid_probs = np.mean(all_probs, axis=0)
    hybrid_preds = hybrid_probs.argmax(axis=1)

    acc = accuracy_score(labels_ref, hybrid_preds)
    f1 = f1_score(labels_ref, hybrid_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(labels_ref, hybrid_preds, weights="quadratic")

    print("\n" + "=" * 50)
    print(f"RESULTADO FINAL -- Ensemble HIBRIDO ({len(softmax_ckpts)} softmax + {len(coral_ckpts)} CORAL) + TTA")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print("\nRelatorio por classe:")
    print(classification_report(labels_ref, hybrid_preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
