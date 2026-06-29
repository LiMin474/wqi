from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(r"D:\study\project\water-quality\Stacking-WQI-Hyperopt\results\method_framework_polished.png")


def add_box(ax, x, y, w, h, text, fc="#F7FAFC", ec="#5B7C99", lw=1.6, fs=13, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        color="#1F2937",
        linespacing=1.4,
    )
    return patch


def add_group(ax, x, y, w, h, title, fc="#FCFCFD", ec="#B9C7D6", fs=15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.035,
        title,
        ha="left",
        va="center",
        fontsize=fs,
        weight="bold",
        color="#243B53",
    )
    return patch


def add_arrow(ax, start, end, rad=0.0, color="#4B5563", lw=1.4):
    arr = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Group containers
    add_group(ax, 0.04, 0.57, 0.22, 0.31, "Stage 1  Data Preparation")
    add_group(ax, 0.31, 0.49, 0.64, 0.39, "Stage 2  Hyperparameter Optimization via Six Evolutionary Algorithms")
    add_group(ax, 0.22, 0.08, 0.56, 0.27, "Stage 3  Ensemble Construction and Final Evaluation")

    # Left stage
    add_box(
        ax,
        0.075,
        0.71,
        0.15,
        0.10,
        "Three WQI Datasets\nJajpur-Groundwater | Irish-River-CCME | AKH-WQI",
        fc="#EEF4FF",
        ec="#7C8DB5",
        fs=12,
        weight="bold",
    )
    add_box(
        ax,
        0.075,
        0.61,
        0.15,
        0.08,
        "Preprocessing and Split\n8:2 hold-out + 5-fold CV",
        fc="#F5F7FA",
        ec="#7C8DB5",
        fs=12,
    )
    add_arrow(ax, (0.15, 0.71), (0.15, 0.69))

    # Optimization header
    add_box(
        ax,
        0.50,
        0.78,
        0.18,
        0.09,
        "ANN Hyperparameter Search\nvia Six Evolutionary Optimizers",
        fc="#EAF6F2",
        ec="#4E8B7A",
        fs=13,
        weight="bold",
    )

    algos = ["DE", "SHADE", "CMA-ES", "NRBO", "BOA", "HHO-Lite"]
    xs = [0.37, 0.47, 0.57, 0.67, 0.77, 0.87]
    for x, algo in zip(xs, algos):
        add_box(ax, x - 0.025, 0.62, 0.05, 0.055, algo, fc="#F3F0FF", ec="#8B7FBF", fs=12, weight="bold")

    add_box(
        ax,
        0.50,
        0.47,
        0.18,
        0.07,
        "Six optimized ANN base models",
        fc="#F5F7FA",
        ec="#5B7C99",
        fs=12,
        weight="bold",
    )

    add_arrow(ax, (0.225, 0.65), (0.31, 0.65), rad=0.0)
    for x in xs:
        add_arrow(ax, (0.59, 0.78), (x, 0.675), rad=0.10 if x < 0.59 else (-0.10 if x > 0.59 else 0.0))
        add_arrow(ax, (x, 0.62), (0.59, 0.54), rad=0.14 if x < 0.59 else (-0.14 if x > 0.59 else 0.0))

    # Bottom stage
    add_box(
        ax,
        0.30,
        0.16,
        0.22,
        0.11,
        "WeightedAvg Ensemble\nWeights derived from normalized R²CV",
        fc="#FFF8E8",
        ec="#C79B2C",
        fs=13,
        weight="bold",
    )
    add_box(
        ax,
        0.57,
        0.16,
        0.16,
        0.11,
        "Final Ensemble Prediction\nMetrics: R², R²CV, RMSE, MAE, Time",
        fc="#EEF4FF",
        ec="#5B7C99",
        fs=12,
        weight="bold",
    )

    add_arrow(ax, (0.59, 0.47), (0.41, 0.27))
    add_arrow(ax, (0.52, 0.215), (0.57, 0.215))

    ax.text(
        0.5,
        0.03,
        "Framework of the proposed multi-evolutionary-algorithm ANN optimization and R²CV-weighted ensemble strategy",
        ha="center",
        va="center",
        fontsize=12,
        color="#52606D",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(str(OUT))


if __name__ == "__main__":
    main()
