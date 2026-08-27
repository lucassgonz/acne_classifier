"""
Baixa e organiza um subconjunto do SCIN (Skin Condition Image Network,
Google Research / Stanford Medicine) para pre-treino supervisionado do
backbone ResNet18. Licenca: SCIN Data Use License (permite pesquisa
academica e treino de modelos, com atribuicao).
https://github.com/google-research-datasets/scin

O SCIN nao tem gravidade de acne — serve apenas para o backbone aprender
features de pele em geral antes do fine-tuning no ACNE04 (unico dataset
com os 4 niveis de gravidade usados neste trabalho).

Uso:
    cd training
    python utils/prepare_scin_pretrain.py --output ../../data/scin_pretrain --min-samples 30
"""
import argparse
import ast
import csv
import os
import urllib.request
from collections import Counter

BUCKET_BASE = "https://storage.googleapis.com/dx-scin-public-data/"
CASES_URL = BUCKET_BASE + "dataset/scin_cases.csv"
LABELS_URL = BUCKET_BASE + "dataset/scin_labels.csv"


def download_file(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_metadata(meta_dir):
    cases_path = os.path.join(meta_dir, "scin_cases.csv")
    labels_path = os.path.join(meta_dir, "scin_labels.csv")
    download_file(CASES_URL, cases_path)
    download_file(LABELS_URL, labels_path)

    with open(cases_path) as f:
        cases = {r["case_id"]: r for r in csv.DictReader(f)}
    with open(labels_path) as f:
        labels = {r["case_id"]: r for r in csv.DictReader(f)}
    return cases, labels


def top1_condition(raw):
    """Extrai a condicao dermatologica de maior confianca (primeiro item da lista)."""
    if not raw or raw == "[]":
        return None
    try:
        conditions = ast.literal_eval(raw)
        return conditions[0] if conditions else None
    except (ValueError, SyntaxError):
        return None


def safe_dirname(name):
    return name.strip().replace(" ", "_").replace(",", "").replace("/", "-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../data/scin_pretrain")
    parser.add_argument("--meta-dir", default="../../data/scin_meta")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="Descarta classes com menos de N casos rotulados")
    args = parser.parse_args()

    print("Baixando metadados do SCIN...")
    cases, labels = load_metadata(args.meta_dir)
    print(f"  {len(cases)} casos no total")

    top1 = {}
    for cid, lab in labels.items():
        cond = top1_condition(lab.get("dermatologist_skin_condition_on_label_name", ""))
        if cond:
            top1[cid] = cond

    counts = Counter(top1.values())
    keep_classes = {c for c, n in counts.items() if n >= args.min_samples}
    print(f"  {len(keep_classes)} classes com >= {args.min_samples} casos:")
    for c in sorted(keep_classes):
        print(f"    {c}: {counts[c]}")

    tasks = []
    for cid, cond in top1.items():
        if cond not in keep_classes:
            continue
        case = cases.get(cid)
        if not case:
            continue
        for i in (1, 2, 3):
            rel_path = case.get(f"image_{i}_path")
            if rel_path:
                url = BUCKET_BASE + rel_path
                dest = os.path.join(args.output, safe_dirname(cond), f"{cid}_{i}.png")
                tasks.append((url, dest))

    print(f"\nBaixando {len(tasks)} imagens para {args.output} ...")
    ok, failed = 0, 0
    for i, (url, dest) in enumerate(tasks, 1):
        if os.path.exists(dest):
            ok += 1
            continue
        try:
            download_file(url, dest)
            ok += 1
        except Exception as e:
            failed += 1
        if i % 200 == 0:
            print(f"  {i}/{len(tasks)} ({ok} ok, {failed} falhas)")

    print(f"\nConcluído: {ok} imagens baixadas, {failed} falhas.")
    print(f"Classes disponíveis para pré-treino em {args.output}/<classe>/")


if __name__ == "__main__":
    main()
