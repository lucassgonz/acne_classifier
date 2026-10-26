"""
A partir das predicoes por CROP ja salvas (fold*.npz de uma config do
run_kfold_clean.py), funde as predicoes das ate-5 regioes de uma MESMA
foto (softmax medio) para obter UMA predicao por foto, e compara com a
acuracia por-crop isolada. Testa a hipotese: o rotulo de um crop de
regiao e o rotulo da foto INTEIRA (nao ha anotacao por regiao), entao
fundir as regioes deveria recuperar parte da acuracia perdida por crop
isolado nao ter contexto suficiente pra evidenciar o grau da foto toda.

Uso:
    cd training
    python analyze_region_fusion.py --config embed_ablation_v3_with
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results/kfold")
    args = ap.parse_args()

    per_crop_labels, per_crop_preds = [], []
    fused_labels, fused_preds, n_regions_used = [], [], []

    for fold in range(5):
        p = Path(args.results_dir) / args.config / f"fold{fold}.npz"
        if not p.exists():
            print(f"fold {fold}: {p} nao encontrado, pulando")
            continue
        d = np.load(p, allow_pickle=True)
        names, labels, probs = d["names"], d["labels"], d["probs"]

        per_crop_labels.extend(labels.tolist())
        per_crop_preds.extend(probs.argmax(axis=1).tolist())

        by_name = defaultdict(list)
        for name, label, prob in zip(names, labels, probs):
            by_name[name].append((label, prob))

        for name, rows in by_name.items():
            lbls = [r[0] for r in rows]
            assert len(set(lbls)) == 1, f"rotulo inconsistente entre regioes de {name}"
            fused_prob = np.mean([r[1] for r in rows], axis=0)
            fused_labels.append(lbls[0])
            fused_preds.append(int(fused_prob.argmax()))
            n_regions_used.append(len(rows))

    per_crop_acc = accuracy_score(per_crop_labels, per_crop_preds)
    per_crop_f1 = f1_score(per_crop_labels, per_crop_preds, average="macro")
    per_crop_qwk = cohen_kappa_score(per_crop_labels, per_crop_preds, weights="quadratic")

    fused_acc = accuracy_score(fused_labels, fused_preds)
    fused_f1 = f1_score(fused_labels, fused_preds, average="macro")
    fused_qwk = cohen_kappa_score(fused_labels, fused_preds, weights="quadratic")

    print(f"=== {args.config} ===")
    print(f"Por-crop (n={len(per_crop_labels)}):  acc={per_crop_acc*100:.2f}%  macroF1={per_crop_f1*100:.2f}%  QWK={per_crop_qwk:.3f}")
    print(f"Fundido por foto (n={len(fused_labels)}, media de {np.mean(n_regions_used):.2f} regioes/foto):  "
          f"acc={fused_acc*100:.2f}%  macroF1={fused_f1*100:.2f}%  QWK={fused_qwk:.3f}")
    print(f"Ganho da fusao: {(fused_acc-per_crop_acc)*100:+.2f}pp acc, {(fused_qwk-per_crop_qwk):+.3f} QWK")


if __name__ == "__main__":
    main()
