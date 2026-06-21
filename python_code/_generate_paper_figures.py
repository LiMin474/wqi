"""
Generate 7 figures for paper
Data source: datasets/results/ directory
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

# Color scheme
COLORS = {
    'DE': '#E74C3C',      # Red
    'SHADE': '#F39C12',   # Orange
    'APSM-jSO': '#9B59B6', # Purple
    'CMA-ES': '#27AE60',  # Green
    'NRBO': '#3498DB',    # Blue
    'Bayesian': '#95A5A6' # Gray
}

RESULTS_DIR = 'datasets/results'
SAVE_DIR = 'datasets/results/figures'

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 图1：各算法R²CV柱状图
# ============================================================
def plot_r2cv_bar():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'r2cv_bar_data.csv'))

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = df['Dataset'].tolist()
    # 使用CSV中实际存在的列名
    algos = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'Bayesian']

    x = np.arange(len(datasets))
    width = 0.15

    for i, algo in enumerate(algos):
        if algo in df.columns:
            values = df[algo].tolist()
            bars = ax.bar(x + i * width, values, width, label=algo, color=COLORS.get(algo, '#95A5A6'))
            # 标注数值
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('R²CV')
    ax.set_title('R²CV Comparison across Datasets')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right')
    ax.set_ylim(0.7, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig1_r2cv_bar.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 1: R2CV bar chart saved")

# ============================================================
# 图2：集成增益散点图（论文核心图）
# ============================================================
def plot_ensemble_gain():
    # 读取单算法结果
    with open(os.path.join(RESULTS_DIR, 'all_results_v2.json'), 'r') as f:
        all_results = json.load(f)

    # 读取集成结果
    with open(os.path.join(RESULTS_DIR, 'new_ensemble_results.json'), 'r') as f:
        ensemble_results = json.load(f)

    # 提取AKH数据集的R²CV
    akh_single = {
        'DE': all_results['4_akh_wqi']['DE']['R2CV'],
        'SHADE': all_results['4_akh_wqi']['SHADE']['R2CV'],
        'APSM-jSO': all_results['4_akh_wqi']['APSM-jSO (2023)']['R2CV'],
        'CMA-ES': all_results['4_akh_wqi']['CMA-ES']['R2CV'],
        'Bayesian': all_results['4_akh_wqi']['Bayesian']['R2CV'],
    }

    akh_ensemble = {
        'SimpleAvg': ensemble_results['4_akh_wqi']['SimpleAvg']['R2CV'],
        'WeightedAvg': ensemble_results['4_akh_wqi']['WeightedAvg']['R2CV'],
        'LRStacking': ensemble_results['4_akh_wqi']['LRStacking']['R2CV'],
        'RidgeStacking': ensemble_results['4_akh_wqi']['RidgeStacking']['R2CV'],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # 单算法点
    x_single = range(5)
    y_single = list(akh_single.values())
    colors_single = [COLORS[k] for k in akh_single.keys()]

    scatter1 = ax.scatter(x_single, y_single, s=100, c=colors_single, marker='o',
                          label='Single Algorithm', zorder=5)

    # 集成方法点
    x_ensemble = range(5, 9)
    y_ensemble = list(akh_ensemble.values())
    colors_ensemble = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']

    scatter2 = ax.scatter(x_ensemble, y_ensemble, s=150, c=colors_ensemble, marker='s',
                          label='Ensemble Method', zorder=5)

    # 水平虚线：最优单算法
    best_single = max(y_single)
    ax.axhline(y=best_single, color='gray', linestyle='--', linewidth=1.5,
               label=f'Best Single ({best_single:.4f})')

    # 标注关键点
    ax.annotate(f'APSM-jSO\n{akh_single["APSM-jSO"]:.4f}',
                xy=(2, akh_single['APSM-jSO']), xytext=(2, 0.72),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate(f'LRStacking\n{akh_ensemble["LRStacking"]:.4f}',
                xy=(7, akh_ensemble['LRStacking']), xytext=(7, 0.92),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # X轴标签
    all_labels = list(akh_single.keys()) + list(akh_ensemble.keys())
    ax.set_xticks(range(9))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')

    ax.set_xlabel('Method')
    ax.set_ylabel('R²CV')
    ax.set_title('Ensemble Gain on AKH Dataset (R²CV: 0.7570 → 0.8843, +12.73%)')
    ax.legend(loc='upper left')
    ax.set_ylim(0.70, 0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig2_ensemble_gain.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 2: Ensemble gain scatter plot saved")

# ============================================================
# 图3：收敛曲线（4个数据集）
# ============================================================
def plot_convergence():
    datasets = ['Jajpur', 'WQI', 'Sample', 'AKH']
    # 使用CSV中实际存在的列名
    algos = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'Bayesian']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, ds in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        try:
            df = pd.read_csv(os.path.join(RESULTS_DIR, f'convergence_{ds}.csv'))

            for algo in algos:
                if algo in df.columns:
                    ax.plot(df['Generation'], df[algo], label=algo,
                            color=COLORS.get(algo, '#95A5A6'), linewidth=1.5)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Data not found for {ds}', ha='center', va='center')

        ax.set_xlabel('Generation')
        ax.set_ylabel('R²CV')
        ax.set_title(f'{ds} Dataset')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig3_convergence.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 3: Convergence curves saved")

# ============================================================
# 图4：帕累托散点图
# ============================================================
def plot_pareto():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'pareto_chart.csv'))

    # 只画5种核心算法
    algos = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'Bayesian']
    df_filtered = df[df['Algorithm'].isin(algos)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algos:
        algo_df = df_filtered[df_filtered['Algorithm'] == algo]
        if len(algo_df) > 0:
            ax.scatter(algo_df['Time(s)'], algo_df['R2CV'],
                       s=80, c=COLORS.get(algo, '#95A5A6'), label=algo, marker='o')

    ax.set_xscale('log')
    ax.set_xlabel('Time (s) - Log Scale')
    ax.set_ylabel('R²CV')
    ax.set_title('Pareto Chart: Accuracy vs Efficiency')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig4_pareto.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 4: Pareto scatter plot saved")

# ============================================================
# 图5：预测vs实际散点图（AKH数据集）
# ============================================================
def plot_prediction_scatter():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'scatter_AKH.csv'))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Normal samples
    normal = df[df['IsDifficult'] == False]
    ax.scatter(normal['Actual'], normal['Ensemble_Pred'],
               s=30, c='#3498DB', alpha=0.6, label='Normal Samples')

    # Difficult samples
    difficult = df[df['IsDifficult'] == True]
    ax.scatter(difficult['Actual'], difficult['Ensemble_Pred'],
               s=50, c='#E74C3C', alpha=0.8, label='Difficult Samples')

    # 对角线 Y=X
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='Y=X Reference')

    ax.set_xlabel('Actual WQI')
    ax.set_ylabel('Predicted WQI')
    ax.set_title('Prediction vs Actual (AKH Dataset)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig5_prediction_scatter.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 5: Prediction scatter plot saved")

# ============================================================
# 图6：算法分歧度热图（AKH数据集）
# ============================================================
def plot_correlation_heatmap():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'correlation_matrix_AKH.csv'), index_col=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=0.9, vmax=1.0, center=0.95,
                square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'})

    ax.set_title('Algorithm Divergence on AKH Dataset (Correlation Matrix)')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig6_correlation_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 6: Correlation heatmap saved")

# ============================================================
# 图7：WeightedAvg权重柱状图
# ============================================================
def plot_weights_distribution():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'weights_distribution.csv'))

    # 转换为宽格式
    df_wide = df.pivot(index='Dataset', columns='Method', values='Weight')

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = df_wide.index.tolist()
    algos = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'NRBO']

    x = np.arange(len(datasets))
    width = 0.15

    for i, algo in enumerate(algos):
        if algo in df_wide.columns:
            values = df_wide[algo].tolist()
            bars = ax.bar(x + i * width, values, width, label=algo,
                          color=COLORS.get(algo, '#95A5A6'))

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Weight')
    ax.set_title('WeightedAvg Weight Distribution')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right')
    ax.set_ylim(0.15, 0.25)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加参考线：均匀权重0.2
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1,
               label='Equal Weight (0.2)')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_weights_distribution.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 7: Weight distribution bar chart saved")


# ============================================================
# 图8：特征相关性热图（4个数据集）
# ============================================================
def plot_correlation_heatmaps():
    datasets = ['Jajpur', 'WQI', 'Sample', 'AKH']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, ds in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        try:
            df = pd.read_csv(os.path.join(RESULTS_DIR, f'correlation_matrix_{ds}.csv'), index_col=0)

            sns.heatmap(df, annot=True, fmt='.2f', cmap='coolwarm',
                        vmin=-1, vmax=1, center=0,
                        square=True, linewidths=0.3,
                        ax=ax, cbar_kws={'shrink': 0.5},
                        annot_kws={'fontsize': 7})
            ax.set_title(f'{ds} Dataset - Feature Correlation')
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Data not found for {ds}', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_correlation_heatmaps.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 8: Correlation heatmaps (4 datasets) saved")


# ============================================================
# 图9：MAE对比箱线图（单算法 vs 集成）
# ============================================================
def plot_mae_boxplot():
    # Read single algorithm results
    with open(os.path.join(RESULTS_DIR, 'all_results_v2.json'), 'r') as f:
        all_results = json.load(f)

    # Read ensemble results
    with open(os.path.join(RESULTS_DIR, 'new_ensemble_results.json'), 'r') as f:
        ensemble_results = json.load(f)

    datasets = ['1_jajpur', '2_wqi_dataset', '3_sample_dataset', '4_akh_wqi']
    dataset_names = {'1_jajpur': 'Jajpur', '2_wqi_dataset': 'WQI',
                     '3_sample_dataset': 'Sample', '4_akh_wqi': 'AKH'}

    algos = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'NRBO']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, ds_key in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        ds_name = dataset_names[ds_key]

        # Single algorithm MAE
        single_mae = []
        for algo in algos:
            if algo in all_results[ds_key]:
                mae = all_results[ds_key][algo].get('MAE', 0)
                single_mae.append(mae)

        # Ensemble MAE
        ensemble_mae = [
            ensemble_results[ds_key]['SimpleAvg']['MAE'],
            ensemble_results[ds_key]['WeightedAvg']['MAE'],
            ensemble_results[ds_key]['LRStacking']['MAE'],
            ensemble_results[ds_key]['RidgeStacking']['MAE']
        ]

        # Combine for boxplot
        data_to_plot = single_mae + ensemble_mae
        labels = algos + ['SimpleAvg', 'WeightedAvg', 'LRStacking', 'RidgeStacking']
        colors = [COLORS.get(a, '#95A5A6') for a in algos] + ['#2C3E50'] * 4

        bp = ax.boxplot([single_mae, ensemble_mae],
                        positions=[1, 2],
                        widths=0.6,
                        patch_artist=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('#3498DB')
        bp['boxes'][1].set_facecolor('#E74C3C')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Single Algorithms\n(5 algorithms)', 'Ensemble Methods\n(4 methods)'])
        ax.set_ylabel('MAE')
        ax.set_title(f'{ds_name} Dataset - MAE Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend with min/max values
        single_min = min(single_mae)
        single_max = max(single_mae)
        ensemble_min = min(ensemble_mae)
        ensemble_max = max(ensemble_mae)

        ax.annotate(f'Single range:\n{min(single_mae):.3f}-{max(single_mae):.3f}',
                    xy=(1, max(single_mae)), xytext=(1.3, max(single_mae)*1.1),
                    fontsize=8, ha='center')
        ax.annotate(f'Ensemble range:\n{min(ensemble_mae):.3f}-{max(ensemble_mae):.3f}',
                    xy=(2, max(ensemble_mae)), xytext=(2.3, max(ensemble_mae)*1.1),
                    fontsize=8, ha='center')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig9_mae_boxplot.png'), bbox_inches='tight')
    plt.close()
    print("[OK] Figure 9: MAE comparison boxplot saved")


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
    print("=" * 50)
    print("Generating 7+ figures for paper...")
    print("=" * 50)

    plot_r2cv_bar()
    plot_ensemble_gain()
    plot_convergence()
    plot_pareto()
    plot_prediction_scatter()
    plot_correlation_heatmap()  # AKH only
    plot_weights_distribution()

    # New figures
    plot_correlation_heatmaps()  # All 4 datasets
    plot_mae_boxplot()           # MAE comparison

    print("=" * 50)
    print(f"[DONE] All figures saved to: {SAVE_DIR}")
    print("=" * 50)