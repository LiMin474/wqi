"""
Generate 7 figures for AKH-WQI Surface Water dataset
Target: 6 evolutionary algorithms + WeightedAvg ensemble
Output: 3_results/figures/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'python_code', 'results')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Fixtures ──
plt.rcParams.update({
    'font.family': 'Arial', 'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

ALGOS = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
COLORS = {
    'DE': '#E74C3C', 'SHADE': '#FFC107', 'CMA-ES': '#27AE60',
    'NRBO': '#3498DB', 'BOA': '#9B59B6', 'HHO-Lite': '#E91E63',
}

AKH_FEAT_NAMES = ['PH', 'Temp', 'Turbidity', 'TSS', 'BOD5',
                  'COD', 'DO', 'Amoni', 'Phosphat', 'Coliforms']

with open(os.path.join(RESULTS_DIR, 'unified_ensemble_results.json'), 'r') as f:
    DATA = json.load(f)['AKH']


def save_fig(name):
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'{name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f'[OK] {name}')


# ════════════════════════════════════════════════
# Figure 1: R²CV bar chart
# ════════════════════════════════════════════════
def plot_fig1():
    global fig
    fig, ax = plt.subplots(figsize=(10, 6))

    single = [DATA['single_results'][a]['R2CV'] for a in ALGOS]
    best_single = max(single)
    wa = DATA['ensemble_results']['WeightedAvg']['R2CV']

    bars = ax.bar(ALGOS, single, width=0.6, color=[COLORS[a] for a in ALGOS],
                  alpha=0.85, edgecolor='black', linewidth=0.5)

    best_idx = single.index(best_single)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)

    ax.axhline(y=wa, color='#E74C3C', linestyle='-', linewidth=2.5,
               label=f'WeightedAvg (R²CV={wa:.4f})')
    ax.annotate(f'R²CV={wa:.4f}', xy=(len(ALGOS)-0.5, wa),
                xytext=(5, -10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#C0392B')
    ax.axhline(y=best_single, color='gray', linestyle='--', linewidth=1.5,
               label=f'Best Single: {ALGOS[best_idx]} (R²CV={best_single:.4f})')

    for bar, val in zip(bars, single):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    improvement = (wa - best_single) / best_single * 100
    ax.annotate(f'+{improvement:.2f}%',
                xy=(len(ALGOS)-1, (wa + best_single) / 2),
                fontsize=14, fontweight='bold', color='#C0392B', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.85))

    ax.set_ylabel('R²CV', fontsize=12)
    ax.set_ylim(0.65, 0.90)
    ax.set_title('Single Algorithm R²CV vs WeightedAvg Ensemble (AKH-WQI Surface Water)',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    save_fig('fig1_r2cv_bar')


# ════════════════════════════════════════════════
# Figure 2: Convergence curves
# ════════════════════════════════════════════════
def plot_fig_conv():
    global fig
    fig, ax_main = plt.subplots(figsize=(10, 6))

    conv = pd.read_csv(os.path.join(RESULTS_DIR, 'convergence_AKH.csv'))

    MAX_X = 15
    conv_order = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    def pad_to_15(series):
        vals = series.dropna().values
        if len(vals) >= MAX_X:
            return vals[:MAX_X]
        else:
            return np.concatenate([vals, [vals[-1]] * (MAX_X - len(vals))])

    x_15 = np.arange(1, MAX_X + 1)

    for algo in conv_order:
        vals = pad_to_15(conv[algo])
        ax_main.plot(x_15, vals, label=algo,
                     color=COLORS[algo], linewidth=1.5, marker='o', markersize=2)
    ax_main.set_xlabel('Evaluation', fontsize=10)
    ax_main.set_ylabel('R²CV', fontsize=10)
    ax_main.set_xlim(1, MAX_X)
    ax_main.set_ylim(0.30, 0.90)
    ax_main.set_title('Convergence Curves on AKH Dataset (6 Evolutionary Algorithms)',
                      fontsize=12)
    ax_main.legend(loc='lower right', fontsize=7, ncol=2)
    ax_main.grid(True, alpha=0.3)

    save_fig('fig2_convergence')


# ════════════════════════════════════════════════
# Figure 3: Pareto chart
# ════════════════════════════════════════════════
def plot_fig_pareto():
    global fig
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in ALGOS:
        r = DATA['single_results'][algo]
        ax.scatter(r['Time'], r['R2CV'], s=130, c=COLORS[algo],
                   edgecolors='black', linewidth=0.5, zorder=5, alpha=0.9)
        ax.annotate(algo, (r['Time'], r['R2CV']),
                    fontsize=9, ha='center', va='bottom', xytext=(0, 6),
                    textcoords='offset points', fontweight='bold')

    wa = DATA['ensemble_results']['WeightedAvg']['R2CV']
    ax.scatter(0.5, wa, s=200, marker='*', c='#E74C3C', edgecolors='black',
               linewidth=0.5, zorder=10, label=f'WeightedAvg (R²CV={wa:.4f})')
    ax.axhline(y=wa, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.4)
    ax.annotate(f'WeightedAvg\n(t~0s)', xy=(0.5, wa),
                fontsize=9, fontweight='bold', color='#C0392B',
                ha='left', va='bottom', xytext=(10, 0),
                textcoords='offset points')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('R²CV', fontsize=11)
    ax.set_xlim(0, None)
    ax.set_ylim(0.65, 0.90)
    ax.set_title('Accuracy vs Computational Efficiency (AKH-WQI Surface Water)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig('fig3_pareto')


# ════════════════════════════════════════════════
# Figure 4: Prediction scatter
# ════════════════════════════════════════════════
def plot_fig_scatter():
    global fig
    fig, ax = plt.subplots(figsize=(8, 8))

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'scatter_AKH.csv'))

    all_vals = pd.concat([df['Actual'], df['WeightedAvg_Pred']])
    margin = (all_vals.max() - all_vals.min()) * 0.08
    lo, hi = all_vals.min() - margin, all_vals.max() + margin

    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Y = X (Perfect Fit)')

    ax.scatter(df['Actual'], df['WeightedAvg_Pred'], s=20, c='#3498DB',
               alpha=0.55, edgecolors='black', linewidth=0.3, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual WQI', fontsize=11)
    ax.set_ylabel('WeightedAvg Predicted WQI', fontsize=11)
    ax.set_title('Ensemble Prediction vs Actual WQI (AKH, 657 Samples)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    save_fig('fig4_prediction_scatter')


# ════════════════════════════════════════════════
# Figure 5: Algorithm correlation heatmap
# ════════════════════════════════════════════════
def plot_fig_algo_corr():
    global fig
    fig, ax = plt.subplots(figsize=(8, 6))

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'correlation_matrix_AKH.csv'), index_col=0)

    vmin = min(df.min().min(), 0.90)
    center = (vmin + 1.0) / 2
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdBu_r',
                vmin=vmin, vmax=1.0, center=center,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson r'})

    ax.set_title('Algorithm Prediction Correlation (AKH Dataset)', fontsize=12)
    save_fig('fig5_correlation_algo')


# ════════════════════════════════════════════════
# Figure 6: Weight distribution
# ════════════════════════════════════════════════
def plot_fig_weights():
    global fig
    fig, ax = plt.subplots(figsize=(10, 5))

    r2cvs = [DATA['single_results'][a]['R2CV'] for a in ALGOS]
    weights = np.array(r2cvs) / np.sum(r2cvs)

    bars = ax.bar(ALGOS, weights, width=0.55, color=[COLORS[a] for a in ALGOS],
                  alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, w in zip(bars, weights):
        ax.annotate(f'{w:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

    ax.axhline(y=1/6, color='gray', linestyle='--', linewidth=1,
               label=f'Equal Weight ({1/6:.4f})')

    ax.set_ylabel('Weight', fontsize=11)
    ax.set_ylim(min(weights) - 0.003, max(weights) + 0.005)
    ax.set_title('WeightedAvg Ensemble Weights (AKH Dataset)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    save_fig('fig6_weights')


# ════════════════════════════════════════════════
# Figure 7: Feature correlation heatmap
# ════════════════════════════════════════════════
def plot_fig_feat_corr():
    global fig
    fig, ax = plt.subplots(figsize=(10, 8))

    data = np.load(os.path.join(RESULTS_DIR, '..', 'datasets', '3_akh_wqi.npz'), allow_pickle=True)
    X = data['X']

    df = pd.DataFrame(X, columns=AKH_FEAT_NAMES)
    df['WQI'] = data['y']

    corr = df.corr(method='pearson')

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8})

    ax.set_title('Feature Correlation Matrix (AKH-WQI, 10 Indicators + WQI)',
                 fontsize=12, pad=15)
    fig.tight_layout()
    save_fig('fig7_feature_correlation')


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 50)
    print('Generating 7 figures for AKH dataset...')
    print('=' * 50)

    plot_fig1()
    plot_fig_conv()
    plot_fig_pareto()
    plot_fig_scatter()
    plot_fig_algo_corr()
    plot_fig_weights()
    plot_fig_feat_corr()

    print('=' * 50)
    print(f'[DONE] All figures saved to: {SAVE_DIR}')
    print('=' * 50)
