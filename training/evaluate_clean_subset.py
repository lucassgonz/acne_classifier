"""
Reavalia ensembles ja treinados separando o teste em imagens com e sem copia
(quase) duplicada no treino/val, usando o relatorio do pHash
(utils/phash_duplicates.py). Nao retreina nada.

Uso:
    cd training
    python evaluate_clean_subset.py --mode region --checkpoints-dir models/ablation_ckpt_clean_swa
    python evaluate_clean_subset.py --mode whole  --checkpoints-dir models/wholeimage_clean_swa
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision import datasets

from evaluate_ensemble_tta import build_tta_transforms, get_device
from run_ablation import AcneDataset, ResNet18WithEmbed, ResNet18NoEmbed


def leaked_test_names(report_path, threshold):
    with open(report_path) as f:
        pairs = json.load(f)["near_pairs"]
    names = set()
    for p in pairs:
        if p["distance"] > threshold or p["split_a"] == p["split_b"]:
            continue
        if p["split_a"] == "test":
            names.add(os.path.basename(p["a"]))
        if p["split_b"] == "test":
            names.add(os.path.basename(p["b"]))
    return names


@torch.no_grad()
def probs_for(model, loader, device, region_aware):
    model.eval()
    out_all = []
    for batch in loader:
        imgs = batch[0].to(device)
        regions = batch[2].to(device) if region_aware else None
        out_all.append(F.softmax(model(imgs, regions), dim=1).cpu().numpy())
    return np.concatenate(out_all)


def report(tag, y, p):
    if len(y) == 0:
        return
    pred = p.argmax(1)
    print(f"  {tag:<28} n={len(y):<5} acc={accuracy_score(y, pred):.4f}  "
          f"F1={f1_score(y, pred, average='weighted', zero_division=0):.4f}  "
          f"QWK={cohen_kappa_score(y, pred, weights='quadratic'):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["region", "whole"], required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--report", default="results/phash_report.json")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--tta-views", type=int, default=7)
    args = parser.parse_args()

    device = get_device()
    data_dir = args.data_dir or ("../data/final" if args.mode == "region" else "../data/wholeimage")
    seeds = [int(s) for s in args.seeds.split(",")]
    prefix = "with_embedding_seed" if args.mode == "region" else "seed"
    ckpts = [os.path.join(args.checkpoints_dir, f"{prefix}{s}.pth") for s in seeds]
    ckpts = [c for c in ckpts if os.path.exists(c)]
    print(f"{len(ckpts)} checkpoints | modo={args.mode} | TTA={args.tta_views}")

    tfs = build_tta_transforms(args.tta_views)

    def make_ds(tf):
        if args.mode == "region":
            return AcneDataset(data_dir, ["test"], tf)
        return datasets.ImageFolder(os.path.join(data_dir, "test"), transform=tf)

    ref_ds = make_ds(tfs[0])
    paths = [s[0] for s in ref_ds.samples]
    y = np.array([s[1] for s in ref_ds.samples])
    names = np.array([os.path.basename(p) for p in paths])

    model_probs = []
    for c in ckpts:
        model = (ResNet18WithEmbed() if args.mode == "region" else ResNet18NoEmbed()).to(device)
        model.load_state_dict(torch.load(c, map_location=device))
        views = [probs_for(model, DataLoader(make_ds(tf), batch_size=16, shuffle=False), device,
                           args.mode == "region") for tf in tfs]
        model_probs.append(np.mean(views, axis=0))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    ens = np.mean(model_probs, axis=0)

    print(f"\nEnsemble ({len(ckpts)} modelos) + TTA:")
    report("teste completo", y, ens)
    for th in (0, 4, 6):
        leaked = leaked_test_names(args.report, th)
        mask = np.array([n in leaked for n in names])
        print(f"\npHash d<={th}: {len(leaked)} imagens de teste com copia em treino/val "
              f"({mask.sum()} amostras)")
        report("somente SEM copia", y[~mask], ens[~mask])
        report("somente COM copia", y[mask], ens[mask])


if __name__ == "__main__":
    main()
