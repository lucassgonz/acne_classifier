"""
Ensemble (varios seeds) + Test-Time Augmentation para o baseline de imagem
inteira (ResNet18NoEmbed, sem recorte por regiao). Mesma tecnica ja usada
em evaluate_ensemble_tta.py para o pipeline region-aware -- aplicada aqui
so na etapa de avaliacao, sem alterar treino/dados.

Uso:
    cd training
    python evaluate_wholeimage_ensemble_tta.py \
        --data-dir ../data/wholeimage \
        --checkpoints-dir models/wholeimage_ckpt \
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
from torchvision import datasets, transforms

from run_ablation import ResNet18NoEmbed, CLASS_NAMES


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
        transforms.RandomRotation((10, 10)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
    for v in variations[: max(0, n_views - 1)]:
        views.append(transforms.Compose(base + [v, transforms.ToTensor()]))

    return views[:n_views]


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs, None)
        probs = F.softmax(out, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/wholeimage")
    parser.add_argument("--checkpoints-dir", default="models/wholeimage_ckpt")
    parser.add_argument("--tta-views", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "seed*.pth")))
    print(f"Checkpoints encontrados: {len(ckpts)}")
    for c in ckpts:
        print(f"  {c}")
    if not ckpts:
        print("Nenhum checkpoint encontrado. Rode run_wholeimage.py com --save-model primeiro.")
        return

    tta_views = build_tta_transforms(args.tta_views)
    print(f"TTA views: {len(tta_views)}")

    all_model_probs = []
    labels_ref = None

    for ckpt_path in ckpts:
        print(f"\nAvaliando {os.path.basename(ckpt_path)}...")
        model = ResNet18NoEmbed().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        view_probs = []
        for i, tf in enumerate(tta_views):
            test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tf)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            probs, labels = predict_probs(model, test_loader, device)
            view_probs.append(probs)
            if labels_ref is None:
                labels_ref = labels
            print(f"  view {i+1}/{len(tta_views)} ok")

        model_avg_probs = np.mean(view_probs, axis=0)
        all_model_probs.append(model_avg_probs)

        preds = model_avg_probs.argmax(axis=1)
        acc = accuracy_score(labels_ref, preds)
        print(f"  Acc (TTA, modelo individual) = {acc:.4f}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    ensemble_probs = np.mean(all_model_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(labels_ref, ensemble_preds)
    f1 = f1_score(labels_ref, ensemble_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(labels_ref, ensemble_preds, weights="quadratic")

    print("\n" + "=" * 50)
    print(f"RESULTADO FINAL -- Imagem inteira -- Ensemble ({len(ckpts)} modelos) + TTA ({len(tta_views)} views)")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print("\nRelatorio por classe:")
    print(classification_report(labels_ref, ensemble_preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
