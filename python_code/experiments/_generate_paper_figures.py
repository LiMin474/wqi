"""
Generate 8 figures for paper
Data source: results/ directory
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set font and style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Color scheme - Updated for 6 algorithms
COLORS = {
    'DE': '#E74C3C',      # Red
    'SHADE': '#F39C12',   # Orange
    'CMA-ES': '#27AE60',  # Green
    'NRBO': '#3498DB',    # Blue
    'BOA': '#9B59B6',     # Purple
    'HHO-Lite': '#E67E22', # Dark Orange
    'Bayesian': '#95A5A6' # Gray (对比方法)
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # python_code/
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
SAVE_DIR = os.path.join(RESULTS_DIR, 'figures')

os.makedirs(SAVE_DIR, exist_ok=True)

DEFAULT_MODEL = 'MLP-lbfgs'

def load_results(model_name=DEFAULT_MODEL):
    path = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
        print(f"[WARNING] comprehensive_results.json not found, falling back to {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    if model_name in data:
        return data[model_name]
    return data

# ============================================================
# 图1：各算法R²CV柱状图 - Updated for 6 algorithms
# ============================================================
def plot_r2cv_bar(model_name=DEFAULT_MODEL):
    results = load_results(model_name)

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = ['Jajpur', 'Irish', 'AKH']
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    x = np.arange(len(datasets))
    width = 0.12

    for i, algo in enumerate(algos):
        values = []
        for ds in datasets:
            if ds in results and algo in results[ds]['single_results']:
                values.append(results[ds]['single_results'][algo]['R2CV'])
            else:
                values.append(0)

        bars = ax.bar(x + i * width, values, width, label=algo, color=COLORS.get(algo, '#95A5A6'))
        # 标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('R²CV')
    ax.set_title(f'{model_name} - R²CV Comparison across Datasets (6 Evolutionary Algorithms)')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig1_r2cv_bar_{model_name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig1_r2cv_bar_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 1: R2CV bar chart saved ({model_name}, 6 algorithms)")

# ============================================================
# 图2：单算法 vs WeightedAvg集成对比柱状图
# ============================================================
def plot_ensemble_gain(model_name=DEFAULT_MODEL):
    results = load_results(model_name)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = ['Jajpur', 'Irish', 'AKH']
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        if ds not in results:
            ax.text(0.5, 0.5, f'Data not found for {ds}', ha='center', va='center')
            continue

        # Single algorithm R²CV
        single_r2cv = []
        for algo in algos:
            if algo in results[ds]['single_results']:
                single_r2cv.append(results[ds]['single_results'][algo]['R2CV'])
            else:
                single_r2cv.append(0)

        # WeightedAvg R²CV
        wa_result = results[ds]['ensemble_results']['WeightedAvg']
        wa_r2cv = wa_result['R2CV'] if isinstance(wa_result, dict) else wa_result

        # Plot single algorithms as bars
        x = np.arange(len(algos))
        colors_single = [COLORS[algo] for algo in algos]
        bars = ax.bar(x, single_r2cv, width=0.6, color=colors_single, alpha=0.8, label='Single Algorithm')

        # Highlight best single algorithm
        best_single_val = max(single_r2cv)
        best_idx = np.argmax(single_r2cv)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2.5)

        # WeightedAvg as a horizontal dashed line
        ax.axhline(y=wa_r2cv, color='#E74C3C', linestyle='-', linewidth=2.5,
                   label=f'WeightedAvg ({wa_r2cv:.4f})')

        # Best single horizontal line
        ax.axhline(y=best_single_val, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Best Single ({best_single_val:.4f})')

        # Annotate improvement percentage
        improvement = (wa_r2cv - best_single_val) / best_single_val * 100
        ax.annotate(f'+{improvement:.2f}%',
                    xy=(len(algos)-0.5, (wa_r2cv + best_single_val) / 2),
                    fontsize=14, fontweight='bold', color='red',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('R²CV')
        ax.set_title(f'{ds} Dataset')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        if ds == 'AKH':
            ax.set_ylim(0.6, 0.85)
        elif ds == 'Irish':
            ax.set_ylim(0.8, 1.01)
        else:
            ax.set_ylim(0.94, 1.01)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig2_ensemble_gain_{model_name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig2_ensemble_gain_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 2: Single vs WeightedAvg comparison saved ({model_name})")

# ============================================================
# 图3：收敛曲线（仅Jajpur数据集，DE/SHADE/CMA-ES/Bayesian）
# ============================================================
def plot_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'convergence_Jajpur.csv'))
        for algo in algos:
            if algo in df.columns:
                ax.plot(df['Generation'], df[algo], label=algo,
                        color=COLORS.get(algo, '#95A5A6'), linewidth=2,
                        marker='o', markersize=3)
        ax.set_xlabel('Generation')
        ax.set_ylabel('R²CV')
        ax.set_title('Jajpur Dataset - Convergence Curves (6 Algorithms)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Convergence data not found', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig3_convergence.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'fig3_convergence.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 3: Convergence curve saved (Jajpur, 6 algorithms)")

# ============================================================
# 图4：帕累托散点图（从JSON直接生成）
# ============================================================
def plot_pareto(model_name=DEFAULT_MODEL):
    results = load_results(model_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = ['Jajpur', 'Irish', 'AKH']
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    markers = ['o', '^', 's']  # Jajpur, Irish, AKH

    for idx, ds in enumerate(datasets):
        if ds not in results:
            continue
        for algo in algos:
            if algo in results[ds]['single_results']:
                r = results[ds]['single_results'][algo]
                ax.scatter(r['Time'], r['R2CV'],
                          s=100, c=COLORS.get(algo, '#95A5A6'),
                          marker=markers[idx], label=f'{ds}' if algo == 'DE' else '',
                          zorder=5, alpha=0.85, edgecolors='black', linewidth=0.5)
                # 标注算法名称（只标Jajpur数据集）
                if ds == 'Jajpur':
                    ax.annotate(algo, (r['Time'], r['R2CV']),
                               fontsize=7, ha='center', va='bottom',
                               xytext=(0, 5), textcoords='offset points')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Jajpur'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Irish'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='AKH')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xscale('log')
    ax.set_xlabel('Time (s) - Log Scale')
    ax.set_ylabel('R²CV')
    ax.set_title(f'{model_name} - Pareto Chart: Accuracy vs Computational Efficiency')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.6, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig4_pareto_{model_name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig4_pareto_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 4: Pareto scatter plot saved ({model_name})")

# ============================================================
# 图5：预测vs实际散点图（Jajpur数据集）
# ============================================================
def plot_prediction_scatter(model_name=DEFAULT_MODEL, dataset='Jajpur'):
    csv_path = os.path.join(RESULTS_DIR, f'scatter_{model_name}_{dataset}.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        csv_path_old = os.path.join(RESULTS_DIR, f'scatter_{dataset}.csv')
        try:
            df = pd.read_csv(csv_path_old)
            print(f"[WARNING] Using old format: {csv_path_old}")
        except FileNotFoundError:
            print(f"[SKIP] Figure 5: {csv_path} not found")
            return

    fig, ax = plt.subplots(figsize=(8, 8))

    all_vals = pd.concat([df['Actual'], df[f'{dataset}_Pred' if f'{dataset}_Pred' in df.columns else 'WeightedAvg_Pred']])
    margin = (all_vals.max() - all_vals.min()) * 0.1
    lo = all_vals.min() - margin
    hi = all_vals.max() + margin

    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Y=X Reference')

    pred_col = f'{dataset}_Pred' if f'{dataset}_Pred' in df.columns else 'WeightedAvg_Pred'
    ax.scatter(df['Actual'], df[pred_col],
               s=30, c='#3498DB', alpha=0.6, edgecolors='black', linewidth=0.3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel('Actual WQI')
    ax.set_ylabel('Predicted WQI')
    ax.set_title(f'{model_name} - Prediction vs Actual ({dataset} Dataset, {len(df)} samples)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig5_prediction_scatter_{model_name}_{dataset}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig5_prediction_scatter_{model_name}_{dataset}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 5: Prediction scatter plot saved ({model_name}, {dataset})")

# ============================================================
# 图6：算法分歧度热图（Jajpur数据集）
# ============================================================
def plot_correlation_heatmap(model_name=DEFAULT_MODEL, dataset='Jajpur'):
    csv_path = os.path.join(RESULTS_DIR, f'correlation_matrix_{dataset}.csv')
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print(f"[SKIP] Figure 6: {csv_path} not found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdBu_r',
                vmin=0.99, vmax=1.0, center=0.995,
                square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'})

    ax.set_title(f'{model_name} - Algorithm Divergence on {dataset} Dataset (Correlation Matrix)')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig6_correlation_heatmap_{model_name}_{dataset}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig6_correlation_heatmap_{model_name}_{dataset}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 6: Correlation heatmap saved ({model_name}, {dataset})")

# ============================================================
# 图7：WeightedAvg权重柱状图 - Updated for 6 algorithms
# ============================================================
def plot_weights_distribution(model_name=DEFAULT_MODEL):
    results = load_results(model_name)

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = ['Jajpur', 'Irish', 'AKH']
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    weights_data = {}
    for ds in datasets:
        if ds not in results:
            weights_data[ds] = {a: 1/6 for a in algos}
            continue
        total_r2cv = sum([results[ds]['single_results'][algo]['R2CV'] for algo in algos])
        weights_data[ds] = {}
        for algo in algos:
            weights_data[ds][algo] = results[ds]['single_results'][algo]['R2CV'] / total_r2cv

    x = np.arange(len(datasets))
    width = 0.12

    for i, algo in enumerate(algos):
        values = [weights_data[ds][algo] for ds in datasets]
        bars = ax.bar(x + i * width, values, width, label=algo,
                      color=COLORS.get(algo, '#95A5A6'))

        # 标注数值
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Weight')
    ax.set_title(f'{model_name} - WeightedAvg Weight Distribution (6 Algorithms)')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0.14, 0.20)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加参考线：均匀权重
    ax.axhline(y=1/6, color='gray', linestyle='--', linewidth=1,
               label='Equal Weight (0.167)')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig7_weights_distribution_{model_name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig7_weights_distribution_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 7: Weight distribution bar chart saved ({model_name})")


# ============================================================
# 图8：特征相关性热图（3个数据集）
# ============================================================
def plot_correlation_heatmaps():
    datasets = ['Jajpur', 'Irish', 'AKH']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        try:
            df = pd.read_csv(os.path.join(RESULTS_DIR, f'correlation_matrix_{ds}.csv'), index_col=0)

            # Per-dataset color range to show meaningful differences
            vmin_map = {'Jajpur': 0.99, 'Irish': 0.95, 'AKH': 0.88}
            center_map = {'Jajpur': 0.995, 'Irish': 0.97, 'AKH': 0.94}
            sns.heatmap(df, annot=True, fmt='.4f', cmap='RdBu_r',
                        vmin=vmin_map[ds], vmax=1.0, center=center_map[ds],
                        square=True, linewidths=0.3,
                        ax=ax, cbar_kws={'shrink': 0.5},
                        annot_kws={'fontsize': 7})
            ax.set_title(f'{ds} Dataset - Algorithm Correlation')
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Data not found for {ds}', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_correlation_heatmaps.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_correlation_heatmaps.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 8: Correlation heatmaps (4 datasets) saved")


# ============================================================
# 图9：MAE对比箱线图（单算法 vs 集成） - Updated for 6 algorithms
# ============================================================
def plot_mae_boxplot(model_name=DEFAULT_MODEL):
    results = load_results(model_name)

    datasets = ['Jajpur', 'Irish', 'AKH']
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        if ds not in results:
            ax.text(0.5, 0.5, f'Data not found for {ds}', ha='center', va='center')
            continue

        # Single algorithm MAE
        single_mae = []
        for algo in algos:
            if algo in results[ds]['single_results']:
                v = results[ds]['single_results'][algo]
                single_mae.append(v['MAE'] if 'MAE' in v else 0)
            else:
                single_mae.append(0)

        # WeightedAvg MAE
        wa = results[ds]['ensemble_results']['WeightedAvg']
        wa_mae = wa['MAE'] if isinstance(wa, dict) else 0

        # Boxplot of 6 single algorithms
        bp = ax.boxplot(single_mae, positions=[1], widths=0.5, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498DB')
        bp['boxes'][0].set_alpha(0.7)

        # WeightedAvg as horizontal line
        ax.axhline(y=wa_mae, color='#E74C3C', linestyle='-', linewidth=2.5,
                   label=f'WeightedAvg (MAE={wa_mae:.3f})')

        # Scatter individual algo MAE with labels
        for i, (algo, mae) in enumerate(zip(algos, single_mae)):
            ax.scatter(1 + np.random.uniform(-0.12, 0.12), mae,
                      color=COLORS[algo], s=50, zorder=5, label=algo if idx == 2 else '')

        ax.set_xticks([1])
        ax.set_xticklabels(['6 Single Algorithms'], fontsize=9)
        ax.set_ylabel('MAE')
        ax.set_title(f'{ds} Dataset - MAE Comparison')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'fig9_mae_boxplot_{model_name}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, f'fig9_mae_boxplot_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 9: MAE boxplot saved ({model_name})")


# ============================================================
# 图10：消融实验 - 去掉某个算法的影响
# ============================================================
def plot_ablation():
    # This would need ablation data - placeholder
    # For now, skip this as it requires running ablation experiments
    print("[SKIP] Figure 10: Ablation study - requires additional experiments")


# ============================================================
# 主函数：生成所有图
# ============================================================
if __name__ == '__main__':
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print("=" * 50)
    print(f"Generating figures for paper (Model: {model_name})...")
    print("=" * 50)

    plot_r2cv_bar(model_name)
    plot_ensemble_gain(model_name)
    # plot_convergence()  # 已跳过：收敛曲线需额外运行实验
    plot_pareto(model_name)
    plot_prediction_scatter(model_name)
    plot_correlation_heatmap(model_name)
    plot_weights_distribution(model_name)

    plot_correlation_heatmaps()  # All 4 datasets
    plot_mae_boxplot(model_name)

    print("=" * 50)
    print(f"[DONE] All figures saved to: {SAVE_DIR}")
    print("=" * 50)