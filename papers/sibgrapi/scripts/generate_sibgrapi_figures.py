#!/usr/bin/env python3
"""Generate SIBGRAPI-118 figures from ablation_metrics.json (publication style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
REGION_LABELS = ["Forehead", "Chin", "Nose", "L. cheek", "R. cheek"]
COLOR_NO = "#8A8F98"
COLOR_WITH = "#1B4F72"
COLOR_GAIN = "#1E7A46"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _annotate_bars(ax, bars, fmt="{:.1f}", dy=2.0, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        # Cap displayed percentages at 99.0 by request (never show 100).
        h_disp = min(h, 99.0) if h >= 0 else h
        ax.annotate(
            fmt.format(h_disp),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def fig_ablation_overall(metrics: dict, out: Path) -> None:
    with_o = metrics["with_embed"]["overall"]
    no_o = metrics["no_embed"]["overall"]
    labels = ["Accuracy", "Weighted F1"]
    with_vals = [with_o["accuracy"] * 100, with_o["f1_weighted"] * 100]
    no_vals = [no_o["accuracy"] * 100, no_o["f1_weighted"] * 100]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    b1 = ax.bar(x - width / 2, no_vals, width, label="NoEmbed (scratch)", color=COLOR_NO, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, with_vals, width, label="WithEmbed", color=COLOR_WITH, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.55, 1.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_ablation_by_region(metrics: dict, out: Path) -> None:
    with_r = metrics["with_embed"]["by_region"]
    no_r = metrics["no_embed"]["by_region"]
    with_vals = [with_r[r]["f1_weighted"] * 100 for r in REGIONS]
    no_vals = [no_r[r]["f1_weighted"] * 100 for r in REGIONS]
    deltas = [w - n for w, n in zip(with_vals, no_vals)]

    x = np.arange(len(REGIONS))
    width = 0.36
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 3.9),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.32},
        constrained_layout=True,
    )

    ax = axes[0]
    b1 = ax.bar(x - width / 2, no_vals, width, label="NoEmbed (scratch)", color=COLOR_NO, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, with_vals, width, label="WithEmbed", color=COLOR_WITH, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_LABELS)
    ax.set_ylabel("Weighted F1 (%)")
    ax.set_ylim(0, 118)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, borderaxespad=0)
    _annotate_bars(ax, b1, fontsize=7, dy=1.5)
    _annotate_bars(ax, b2, fontsize=7, dy=1.5)
    ax.text(0.0, -0.22, "(a) Per-region weighted F1", transform=ax.transAxes, ha="left", fontsize=10)

    ax = axes[1]
    bars = ax.bar(x, deltas, color=COLOR_GAIN, edgecolor="white", linewidth=0.5, width=0.62)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_LABELS)
    ax.set_ylabel("F1 gain (pp)")
    ax.set_ylim(0, max(deltas) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _annotate_bars(ax, bars, fmt="{:+.1f}", fontsize=8, dy=1.5)
    ax.text(0.0, -0.22, "(b) Embedding gain by region", transform=ax.transAxes, ha="left", fontsize=10)

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_embedding_similarity(metrics: dict, out: Path) -> None:
    sim = np.array(metrics["embedding_cosine_similarity"], dtype=float)
    # Never display 1.00 / 100% on the diagonal; cap reporting at 0.99.
    sim_plot = np.clip(sim, -1.0, 0.99)
    np.fill_diagonal(sim_plot, 0.99)
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    sns.heatmap(
        sim_plot,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=-0.20,
        vmax=0.99,
        xticklabels=REGION_LABELS,
        yticklabels=REGION_LABELS,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Cosine similarity", "shrink": 0.85},
        annot_kws={"size": 8},
    )
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc="#EAF0F6", ec="#1B4F72"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    return patch


def _arrow(ax, p1, p2):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.15))


def fig_ablation_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    _box(ax, (0.25, 1.45), 1.85, 1.05, "Facial region\ncrops (224×224)")
    _box(ax, (2.45, 1.45), 1.85, 1.05, "Shared\nResNet-18")
    _box(ax, (4.75, 2.45), 2.05, 0.95, "WithEmbed\n[f ‖ e$_r$] → head", fc="#E4F0E6", ec="#1E7A46")
    _box(ax, (4.75, 0.55), 2.05, 0.95, "NoEmbed\nf → head", fc="#F6E8E8", ec="#A93226")
    _box(ax, (7.45, 2.45), 2.05, 0.95, "4-class\nseverity logits", fc="#E4F0E6", ec="#1E7A46")
    _box(ax, (7.45, 0.55), 2.05, 0.95, "4-class\nseverity logits", fc="#F6E8E8", ec="#A93226")

    _arrow(ax, (2.1, 1.97), (2.45, 1.97))
    _arrow(ax, (4.3, 2.2), (4.75, 2.9))
    _arrow(ax, (4.3, 1.75), (4.75, 1.05))
    _arrow(ax, (6.8, 2.92), (7.45, 2.92))
    _arrow(ax, (6.8, 1.02), (7.45, 1.02))
    ax.text(5.8, 3.55, "identical training protocol", ha="center", fontsize=8, style="italic", color="#444444")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_app_concept(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.8)
    ax.axis("off")

    steps = [
        (0.25, "Capture /\nselect image"),
        (2.65, "Extract facial\nregions"),
        (5.05, "Region-aware\nclassification"),
        (7.45, "Severity report\n& history"),
    ]
    for x, text in steps:
        _box(ax, (x, 0.75), 1.95, 1.25, text, fc="#F7F3EA", ec="#6D4C41")
    for x0, x1 in [(2.2, 2.65), (4.6, 5.05), (7.0, 7.45)]:
        _arrow(ax, (x0, 1.37), (x1, 1.37))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="/Users/GonLu/Documents/pessoal/artigo-118-revisao/results/ablation_metrics.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/GonLu/Documents/pessoal/artigo-118-revisao/figs",
    )
    args = parser.parse_args()
    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_ablation_pipeline(out_dir / "fig_ablation_pipeline.png")
    fig_app_concept(out_dir / "fig_app_concept.png")

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics not found yet: {metrics_path}")
        return

    metrics = load_metrics(metrics_path)
    fig_ablation_overall(metrics, out_dir / "fig_ablation_overall.png")
    fig_ablation_by_region(metrics, out_dir / "fig_ablation_by_region.png")
    fig_embedding_similarity(metrics, out_dir / "fig_embedding_similarity.png")
    print(f"Wrote polished figures to {out_dir}")


if __name__ == "__main__":
    main()
