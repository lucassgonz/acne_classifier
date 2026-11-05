"""Fila de treinos da receita JAGC (ResNet-18), varios processos em paralelo na mesma GPU (Colab ou PC local).

Cada tarefa e um (job, fold); folds com .npz ja salvo no destino sao pulados, entao a fila pode ser
relancada sem perder nada. Uso: python colab_queue.py --out DIR --parallel 3 --jobs off45 off46 ...
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

OFFICIAL = []
COMBINED_ALL = ["--splits", "results/kfold_splits_wholeimage_v2.json",
                "--manifest-v2", "../data/final_wholeimage_v2/manifest.json",
                "--img-dir", "../data/jagc_replica_300_combined"]
COMBINED = COMBINED_ALL + ["--only-source", "a04"]
HIRES = ["--img-dir", "../data/jagc_replica_384", "--resize", "330", "--crop", "288"]

# Uma thread de BLAS/OpenMP por processo: com varios treinos e workers em paralelo, os buffers por thread
# do OpenBLAS esgotam a memoria virtual (Windows) e os workers do DataLoader morrem.
CHILD_ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

JOBS = {}
for s in (45, 46, 47, 48):
    JOBS[f"off{s}"] = (f"jagc_replica_resnet18_seed{s}", s, OFFICIAL + ["--img-dir", "../data/jagc_replica_300"])
for s in (43, 44, 45, 46):
    JOBS[f"a04_{s}"] = (f"jagc_replica_combined_a04only_seed{s}", s, COMBINED)
for s in (45, 46, 47, 48):
    JOBS[f"comb{s}"] = (f"jagc_replica_combined_resnet18_seed{s}", s, COMBINED_ALL)
JOBS["hires42"] = ("jagc_replica_resnet18_hires_seed42", 42, HIRES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--logs", default=os.path.join(tempfile.gettempdir(), "jagc_logs"))
    ap.add_argument("--jobs", nargs="+", required=True, choices=list(JOBS))
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    tasks = []
    for j in args.jobs:
        name, seed, extra = JOBS[j]
        for fold in range(5):
            if os.path.exists(os.path.join(args.out, name, f"fold{fold}.npz")):
                continue
            cmd = [sys.executable, "run_jagc_original_replica.py", "--fold", str(fold), "--arch", "resnet18",
                   "--seed", str(seed), "--name", name, "--tta", "--num-workers", str(args.workers),
                   "--output-dir", args.out] + extra
            tasks.append((f"{name}_fold{fold}", cmd))
    print(f"{len(tasks)} tarefas na fila, {args.parallel} em paralelo", flush=True)

    running = []
    while tasks or running:
        running = [(n, p, t) for n, p, t in running if p.poll() is None]
        while tasks and len(running) < args.parallel:
            n, cmd = tasks.pop(0)
            log = open(os.path.join(args.logs, f"{n}.log"), "w")
            running.append((n, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV), time.time()))
            print(time.strftime("%H:%M:%S"), "iniciou", n, flush=True)
        time.sleep(20)
    print(time.strftime("%H:%M:%S"), "FILA CONCLUIDA", flush=True)


if __name__ == "__main__":
    main()
