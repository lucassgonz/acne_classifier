"""
Avaliacao final combinando Ensemble (varios modelos, uma por seed) e
Test-Time Augmentation (varias visualizacoes aumentadas de cada imagem de
teste, com as predicoes de probabilidade medias entre elas).

Ambas as tecnicas sao aplicadas so no momento de AVALIACAO -- nao alteram
o treino, os dados de treino, nem introduzem qualquer forma de vazamento.
Sao tecnicas padrao e bem estabelecidas para melhorar a estimativa final
de um modelo ja treinado.

Uso:
    cd training
    python evaluate_ensemble_tta.py \
        --data-dir ../data/final \
        --checkpoints-dir models/ablation_ckpt \
        --tag with_embedding \
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

from run_ablation import AcneDataset, ResNet18WithEmbed, ResNet18NoEmbed, REGIONS, CLASS_NAMES


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_tta_transforms(n_views):
    """
    Gera N transformacoes de teste: a original (sem augmentation) sempre
    incluida, mais variacoes leves (flip, pequena rotacao, jitter) para
    capturar robustez a pequenas variacoes de captura -- sem alterar o
    conteudo semantico da imagem.
    """
    base = [transforms.Resize((224, 224))]
    views = [transforms.Compose(base + [transforms.ToTensor()])]  # original

    variations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation((5, 5)),
        transforms.RandomRotation((-5, -5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    for v in variations[: max(0, n_views - 1)]:
        views.append(transforms.Compose(base + [v, transforms.ToTensor()]))

    return views[:n_views]


@torch.no_grad()
def predict_probs(model, loader, device):
    """Retorna probabilidades softmax [N, C] e labels verdadeiros [N]."""
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, regions in loader:
        imgs, regions = imgs.to(device), regions.to(device)
        out = model(imgs, regions)
        probs = F.softmax(out, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/final")
    parser.add_argument("--checkpoints-dir", default="models/ablation_ckpt")
    parser.add_argument("--tag", default="with_embedding", choices=["with_embedding", "no_embedding"])
    parser.add_argument("--tta-views", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, f"{args.tag}_seed*.pth")))
    print(f"Checkpoints encontrados ({args.tag}): {len(ckpts)}")
    for c in ckpts:
        print(f"  {c}")
    if not ckpts:
        print("Nenhum checkpoint encontrado. Rode run_ablation.py com --save-models-dir primeiro.")
        return

    tta_views = build_tta_transforms(args.tta_views)
    print(f"TTA views: {len(tta_views)}")

    all_model_probs = []
    labels_ref = None

    for ckpt_path in ckpts:
        print(f"\nAvaliando {os.path.basename(ckpt_path)}...")
        model_cls = ResNet18WithEmbed if args.tag == "with_embedding" else ResNet18NoEmbed
        model = model_cls().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        view_probs = []
        for i, tf in enumerate(tta_views):
            test_ds = AcneDataset(args.data_dir, ["test"], tf)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            probs, labels = predict_probs(model, test_loader, device)
            view_probs.append(probs)
            if labels_ref is None:
                labels_ref = labels
            print(f"  view {i+1}/{len(tta_views)} ok")

        # Media das probabilidades entre as views de TTA para este modelo
        model_avg_probs = np.mean(view_probs, axis=0)
        all_model_probs.append(model_avg_probs)

        preds = model_avg_probs.argmax(axis=1)
        acc = accuracy_score(labels_ref, preds)
        print(f"  Acc (TTA, modelo individual) = {acc:.4f}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # Ensemble final: media das probabilidades (ja com TTA aplicado) entre todos os modelos
    ensemble_probs = np.mean(all_model_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(labels_ref, ensemble_preds)
    f1 = f1_score(labels_ref, ensemble_preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(labels_ref, ensemble_preds, weights="quadratic")

    print("\n" + "=" * 50)
    print(f"RESULTADO FINAL -- Ensemble ({len(ckpts)} modelos) + TTA ({len(tta_views)} views)")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print("\nRelatorio por classe:")
    print(classification_report(labels_ref, ensemble_preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
