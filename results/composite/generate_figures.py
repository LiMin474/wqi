"""
Generate 4 composite figures combining results from all 3 datasets (Jajpur, Irish, AKH).

Figure 1: R²CV bar chart comparison (3 panels side by side)
Figure 2: Ensemble gain comparison (single figure with grouped bars)
Figure 3: Weights distribution (3 panels side by side, horizontal bars)
Figure 4: Correlation heatmaps (3 panels side by side)
"""

import os
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "python_code", "results")
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Constants
# ----------------------------
DATASETS = ["Jajpur", "Irish", "AKH"]

ALGOS = ["DE", "SHADE", "CMA-ES", "NRBO", "BOA", "HHO-Lite"]

COLORS = {
    "DE": "#1f77b4",
    "SHADE": "#FFC107",
    "CMA-ES": "#2ca02c",
    "NRBO": "#d62728",
    "BOA": "#7b1fa2",
    "HHO-Lite": "#E91E63",
    "WeightedAvg": "#dc143c",
}

# Y-axis limits per dataset for Figure 1
Y_LIMITS = {
    "Jajpur": (0.97, 1.005),
    "Irish": (0.93, 1.005),
    "AKH": (0.65, 0.90),
}

# ----------------------------
# Matplotlib global settings
# ----------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ====================
# Data Loading
# ====================
def load_json_data():
    """Load the unified ensemble results JSON."""
    json_path = os.path.join(RESULTS_DIR, "unified_ensemble_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_weights_csv(dataset):
    """Load weights CSV for a given dataset; returns a dict {algo: weight}."""
    path = os.path.join(RESULTS_DIR, f"weights_{dataset}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return dict(zip(df["Algorithm"], df["Weight"]))


def load_correlation_csv(dataset):
    """Load correlation matrix CSV for a given dataset; returns a DataFrame."""
    path = os.path.join(RESULTS_DIR, f"correlation_matrix_{dataset}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df


# ====================
# Figure 1: R²CV bar chart
# ====================
def plot_figure1(data):
    """3-panel side-by-side bar chart of R²CV for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, dataset in zip(axes, DATASETS):
        ds = data[dataset]
        single_results = ds["single_results"]
        ensemble_results = ds["ensemble_results"]["WeightedAvg"]

        # Gather R²CV values for the 6 single algorithms
        algo_r2cv = [single_results[a]["R2CV"] for a in ALGOS]
        wa_r2cv = ensemble_results["R2CV"]

        # Bar positions
        x = np.arange(len(ALGOS))
        bar_width = 0.55

        bars = ax.bar(x, algo_r2cv, bar_width, color=[COLORS[a] for a in ALGOS], edgecolor="white")

        # WeightedAvg as red dashed horizontal line
        ax.axhline(y=wa_r2cv, color=COLORS["WeightedAvg"], linestyle="--", linewidth=2, label="WeightedAvg")

        # Value label on the WeightedAvg line
        ax.text(
            x[-1] + 0.4,
            wa_r2cv,
            f"{wa_r2cv:.4f}",
            color=COLORS["WeightedAvg"],
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

        # Customize
        ax.set_xlim(-0.5, len(ALGOS) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, fontsize=10, rotation=30, ha="right")
        ax.set_ylabel("R²CV", fontsize=12)
        ax.set_title(dataset, fontsize=14, fontweight="bold")
        ax.set_ylim(Y_LIMITS[dataset])
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("R²CV Comparison: Single Algorithms vs WeightedAvg Ensemble", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_composite_r2cv.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 1 saved: fig_composite_r2cv.[png|pdf]")


# ====================
# Figure 2: Ensemble gain
# ====================
def plot_figure2(data):
    """Grouped bar chart comparing best single algorithm vs WeightedAvg ensemble."""
    fig, ax = plt.subplots(figsize=(8, 5))

    best_singles = []
    ensembles = []
    gain_pcts = []

    for dataset in DATASETS:
        ds = data[dataset]
        ensemble_results = ds["ensemble_results"]["WeightedAvg"]
        wa_r2cv = ensemble_results["R2CV"]
        ensembles.append(wa_r2cv)

        # Find best single algorithm R²CV among the 6 target algos
        single_results = ds["single_results"]
        best_r2cv = max(single_results[a]["R2CV"] for a in ALGOS)
        best_singles.append(best_r2cv)

        gain_pct = (wa_r2cv - best_r2cv) / best_r2cv * 100
        gain_pcts.append(gain_pct)

    x = np.arange(len(DATASETS))
    bar_width = 0.30

    bars1 = ax.bar(x - bar_width / 2, best_singles, bar_width, label="Best Single", color="#2c3e50", edgecolor="white")
    bars2 = ax.bar(x + bar_width / 2, ensembles, bar_width, label="WeightedAvg Ensemble", color="#dc143c", edgecolor="white")

    # Red arrows and gain labels
    for i in range(len(DATASETS)):
        y_bottom = min(best_singles[i], ensembles[i])
        y_top = max(best_singles[i], ensembles[i])
        mid_x = x[i]
        # Arrow
        ax.annotate(
            "",
            xy=(mid_x + 0.15, y_top),
            xytext=(mid_x + 0.15, y_bottom),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        # Gain label
        label_y = (y_bottom + y_top) / 2
        ax.text(
            mid_x + 0.25,
            label_y,
            f"+{gain_pcts[i]:.2f}%",
            color="red",
            fontsize=11,
            fontweight="bold",
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("R²CV", fontsize=12)
    ax.set_title("Ensemble Gain: Best Single vs WeightedAvg", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_composite_gain.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 2 saved: fig_composite_gain.[png|pdf]")


# ====================
# Figure 3: Weights distribution
# ====================
def plot_figure3(weights_dict):
    """3-panel horizontal bar chart of algorithm weights for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Reverse order for horizontal bar (top-to-bottom display)
    algos_rev = list(reversed(ALGOS))

    for ax, dataset in zip(axes, DATASETS):
        w = weights_dict.get(dataset)
        if w is None:
            ax.text(0.5, 0.5, "No weight data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset, fontsize=13, fontweight="bold")
            continue

        vals = [w[a] for a in algos_rev]
        colors_rev = [COLORS[a] for a in algos_rev]

        bars = ax.barh(algos_rev, vals, color=colors_rev, edgecolor="white", height=0.6)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Weight", fontsize=11)
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.25 if vals else 0.25)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Algorithm Weights in WeightedAvg Ensemble", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_composite_weights.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 3 saved: fig_composite_weights.[png|pdf]")


# ====================
# Figure 4: Correlation heatmaps
# ====================
def plot_figure4(corr_dict):
    """3-panel correlation heatmap for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, dataset in zip(axes, DATASETS):
        corr = corr_dict.get(dataset)
        if corr is None:
            ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset, fontsize=13, fontweight="bold")
            continue

        if dataset == "Jajpur":
            # Match standalone Jajpur fig5: tight range, RdBu_r → deep red look
            sns.heatmap(
                corr,
                vmin=0.99, vmax=1.0, center=0.995,
                cmap="RdBu_r",
                annot=True, fmt=".4f",
                square=True, ax=ax,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5,
            )
        else:
            sns.heatmap(
                corr,
                vmin=0.85, vmax=1.0,
                cmap="RdYlBu_r",
                annot=True, fmt=".3f",
                square=True, ax=ax,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5,
            )
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Prediction Correlation (Pearson r) Between Algorithms", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_composite_heatmap.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 4 saved: fig_composite_heatmap.[png|pdf]")


# ====================
# Main
# ====================
def main():
    print("=" * 60)
    print("Generating composite figures for all datasets...")
    print("=" * 60)

    # Load data
    data = load_json_data()
    print(f"  Loaded JSON data for datasets: {list(data.keys())}")

    # Load weights from CSV
    weights_dict = {}
    for ds in DATASETS:
        w = load_weights_csv(ds)
        weights_dict[ds] = w
        if w is not None:
            print(f"  Loaded weights for {ds}: {w}")
        else:
            print(f"  WARNING: No weights CSV for {ds}")

    # Load correlation matrices
    corr_dict = {}
    for ds in DATASETS:
        c = load_correlation_csv(ds)
        corr_dict[ds] = c
        if c is not None:
            print(f"  Loaded correlation matrix for {ds}, shape={c.shape}")
        else:
            print(f"  WARNING: No correlation matrix CSV for {ds}")

    print("\n" + "-" * 60)
    plot_figure1(data)
    plot_figure2(data)
    plot_figure3(weights_dict)
    plot_figure4(corr_dict)
    print("-" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
