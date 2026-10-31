"""Gera as 1457 imagens do split oficial em maior resolucao (padrao 384x384), mesma origem das de 300px."""
import json
import os
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 384
OUT = os.path.join(ROOT, "data", f"jagc_replica_{SIZE}")
os.makedirs(OUT, exist_ok=True)

names = sorted(os.listdir(os.path.join(ROOT, "data", "jagc_replica_300")))
level_dirs = {}
for dp, _, fs in os.walk(os.path.join(ROOT, "Dataset")):
    for f in fs:
        level_dirs[f] = os.path.join(dp, f)

missing = 0
for n in names:
    src = os.path.join(ROOT, "acne_1024", n)
    if not os.path.exists(src):
        src = level_dirs.get(n)
    if src is None:
        missing += 1
        continue
    Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(os.path.join(OUT, n), quality=95)
print(f"{len(names) - missing}/{len(names)} imagens em {OUT}")
