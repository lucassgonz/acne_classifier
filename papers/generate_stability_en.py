import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RUNS = [
    93.76, 93.12, 94.31, 92.57, 93.67, 92.48, 93.12, 93.85, 93.49, 93.30,
    93.30, 92.84, 93.67, 92.75, 93.76, 93.03, 93.39, 93.39, 93.49, 93.21,
    93.30, 92.29, 93.39, 93.21, 93.94, 93.21, 92.57, 92.84, 93.67, 93.30,
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(4.2, 4.8))
sns.boxplot(y=RUNS, width=0.35, color="#4c72b0", ax=ax)
sns.stripplot(y=RUNS, color="red", alpha=0.75, jitter=0.15, ax=ax, label="Individual Runs")

ax.set_ylabel("Accuracy (%)")
ax.set_xticks([])
ax.set_ylim(90, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_title("Distribution of Model Accuracy over 30 Runs", fontsize=11)
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "stability_en.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "stability_en.pdf"), bbox_inches="tight")
print(f"mean={np.mean(RUNS):.2f}, std={np.std(RUNS):.2f}")
