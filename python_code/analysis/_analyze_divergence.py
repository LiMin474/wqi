"""
算法分歧度分析
使用多次分层划分替代单次划分，增强统计显著性检验的可靠性
"""
import numpy as np
import json, os
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt

NUM_SPLITS = 5


def analyze_divergence_single_split(X_train, y_train, X_test, y_test, methods):
    """单次划分的分歧度分析"""
    predictions = {}
    for method_name, method_func in methods.items():
        Mdl, result = method_func(X_train, y_train)
        y_pred = Mdl.predict(X_test)
        predictions[method_name] = y_pred

    method_names = list(predictions.keys())
    n_methods = len(method_names)
    corr_matrix = np.zeros((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(predictions[method_names[i]], predictions[method_names[j]])
                corr_matrix[i, j] = corr

    pred_array = np.array([predictions[m] for m in method_names])
    pred_std = np.std(pred_array, axis=0)
    avg_std = np.mean(pred_std)
    avg_corr = (np.sum(corr_matrix) - n_methods) / (n_methods * (n_methods - 1))

    return {
        'predictions': {k: v.tolist() for k, v in predictions.items()},
        'y_test': y_test.tolist(),
        'correlation_matrix': corr_matrix.tolist(),
        'avg_correlation': avg_corr,
        'avg_std': avg_std,
        'method_names': method_names
    }


def analyze_divergence(ds_name, ds_path, methods):
    """分析算法分歧度（多次划分）"""
    print(f"\n分析 {ds_name}...")

    data = np.load(ds_path)
    X, y = data['X'], data['y']

    all_split_results = []
    all_corr_matrices = []
    all_avg_corr = []
    all_avg_std = []

    for split_idx in range(NUM_SPLITS):
        print(f"  划分 {split_idx + 1}/{NUM_SPLITS}...", end=' ')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + split_idx
        )

        split_result = analyze_divergence_single_split(X_train, y_train, X_test, y_test, methods)
        all_split_results.append(split_result)
        all_corr_matrices.append(split_result['correlation_matrix'])
        all_avg_corr.append(split_result['avg_correlation'])
        all_avg_std.append(split_result['avg_std'])
        print(f"AvgCorr={split_result['avg_correlation']:.4f}, AvgStd={split_result['avg_std']:.4f}")

    overall_corr_matrix = np.mean(all_corr_matrices, axis=0)
    overall_avg_corr = np.mean(all_avg_corr)
    overall_avg_std = np.mean(all_avg_std)
    std_avg_corr = np.std(all_avg_corr)
    std_avg_std = np.std(all_avg_std)

    print(f"\n  分歧度统计 ({NUM_SPLITS}次划分平均):")
    print(f"    平均Pearson相关系数: {overall_avg_corr:.4f} ± {std_avg_corr:.4f}")
    print(f"    平均预测标准差: {overall_avg_std:.4f} ± {std_avg_std:.4f}")
    print(f"    相关系数范围: {overall_corr_matrix.min():.4f} ~ {overall_corr_matrix.max():.4f}")

    print(f"\n  平均Pearson相关系数矩阵:")
    method_names = all_split_results[0]['method_names']
    print("    " + "  ".join([f"{m[:6]:6s}" for m in method_names]))
    for i, m in enumerate(method_names):
        row = "    " + "  ".join([f"{overall_corr_matrix[i,j]:.4f}" for j in range(len(method_names))])
        print(f"{m[:6]:6s} {row}")

    return {
        'dataset': ds_name,
        'num_splits': NUM_SPLITS,
        'overall_correlation_matrix': overall_corr_matrix.tolist(),
        'overall_avg_correlation': overall_avg_corr,
        'overall_avg_std': overall_avg_std,
        'std_avg_corr': std_avg_corr,
        'std_avg_std': std_avg_std,
        'method_names': method_names,
        'split_results': all_split_results
    }


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
    RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

    datasets = {
        'AKH': os.path.join(DATASET_DIR, '3_akh_wqi.npz'),
        'Irish': os.path.join(DATASET_DIR, '2_irish_river.npz'),
        'Jajpur': os.path.join(DATASET_DIR, '1_jajpur.npz')
    }

    methods = {
        'DE': a4_DE_fitrnet_opt,
        'SHADE': a4_SHADE_fitrnet_opt,
        'CMA-ES': a4_CMAES_fitrnet_opt,
        'NRBO': a4_NRBO_fitrnet_opt,
        'BOA': a4_BOA_fitrnet_opt,
        'HHO-Lite': a4_HHO_Lite_fitrnet_opt
    }

    results = {}
    for ds_name, ds_path in datasets.items():
        results[ds_name] = analyze_divergence(ds_name, ds_path, methods)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'divergence_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n分歧度分析结果已保存: {os.path.join(RESULTS_DIR, 'divergence_analysis.json')}")

    for ds_name, result in results.items():
        method_names = result['method_names']
        corr_matrix = result['overall_correlation_matrix']

        with open(os.path.join(RESULTS_DIR, f'correlation_matrix_{ds_name}.csv'), 'w') as f:
            f.write(',' + ','.join(method_names) + '\n')
            for i, m in enumerate(method_names):
                row = m + ',' + ','.join([f"{corr_matrix[i][j]:.4f}" for j in range(len(method_names))])
                f.write(row + '\n')

        print(f"相关系数矩阵CSV已保存: {os.path.join(RESULTS_DIR, f'correlation_matrix_{ds_name}.csv')}")

    summary = []
    for ds_name, result in results.items():
        summary.append({
            'dataset': ds_name,
            'avg_correlation': result['overall_avg_correlation'],
            'std_correlation': result['std_avg_corr'],
            'avg_std': result['overall_avg_std'],
            'std_std': result['std_avg_std'],
            'num_splits': result['num_splits']
        })

    with open(os.path.join(RESULTS_DIR, 'divergence_summary.csv'), 'w') as f:
        f.write('Dataset,AvgCorrelation,StdCorrelation,AvgStd,StdStd,NumSplits\n')
        for s in summary:
            f.write(f"{s['dataset']},{s['avg_correlation']:.4f},{s['std_correlation']:.4f},"
                    f"{s['avg_std']:.4f},{s['std_std']:.4f},{s['num_splits']}\n")

    print(f"分歧度汇总表已保存: {os.path.join(RESULTS_DIR, 'divergence_summary.csv')}")


if __name__ == '__main__':
    main()
