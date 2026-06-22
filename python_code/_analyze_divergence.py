"""
算法分歧度分析
计算6种进化算法预测值之间的Pearson相关系数和标准差
"""
import numpy as np
import json
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt
from common_codes.a4_BOA_fitrnet_opt import a4_BOA_fitrnet_opt
from common_codes.a4_HHO_Lite_fitrnet_opt import a4_HHO_Lite_fitrnet_opt

def analyze_divergence(ds_name, ds_path, methods):
    """分析算法分歧度"""
    print(f"\n分析 {ds_name}...")

    # 加载数据
    data = np.load(ds_path)
    X, y = data['X'], data['y']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练所有模型并获取预测
    predictions = {}
    for method_name, method_func in methods.items():
        print(f"  训练 {method_name}...", end=' ')
        Mdl, result = method_func(X_train, y_train)
        y_pred = Mdl.predict(X_test)
        predictions[method_name] = y_pred
        r2cv = result.get('R2CV', 0)
        print(f"R2CV={r2cv:.4f}")

    # 计算Pearson相关系数矩阵
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

    # 计算预测标准差（分歧度）
    pred_array = np.array([predictions[m] for m in method_names])
    pred_std = np.std(pred_array, axis=0)
    avg_std = np.mean(pred_std)

    # 计算平均相关系数（排除自相关）
    avg_corr = (np.sum(corr_matrix) - n_methods) / (n_methods * (n_methods - 1))

    print(f"\n  分歧度统计:")
    print(f"    平均Pearson相关系数: {avg_corr:.4f}")
    print(f"    平均预测标准差: {avg_std:.4f}")
    print(f"    相关系数范围: {corr_matrix.min():.4f} ~ {corr_matrix.max():.4f}")

    # 输出相关系数矩阵
    print(f"\n  Pearson相关系数矩阵:")
    print("    " + "  ".join([f"{m[:6]:6s}" for m in method_names]))
    for i, m in enumerate(method_names):
        row = "    " + "  ".join([f"{corr_matrix[i,j]:.4f}" for j in range(n_methods)])
        print(f"{m[:6]:6s} {row}")

    return {
        'dataset': ds_name,
        'correlation_matrix': corr_matrix.tolist(),
        'method_names': method_names,
        'avg_correlation': avg_corr,
        'avg_std': avg_std,
        'predictions': {k: v.tolist() for k, v in predictions.items()},
        'y_test': y_test.tolist()
    }

def main():
    datasets = {
        'AKH': 'datasets/4_akh_wqi.npz',
        'WQI': 'datasets/2_wqi_dataset.npz',
        'Sample': 'datasets/3_sample_dataset.npz',
        'Jajpur': 'datasets/1_jajpur.npz'
    }

    # 六个进化算法
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

    # 保存结果
    with open('datasets/results/divergence_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n分歧度分析结果已保存: datasets/results/divergence_analysis.json")

    # 生成CSV格式的相关系数矩阵（用于Origin画热图）
    for ds_name, result in results.items():
        method_names = result['method_names']
        corr_matrix = result['correlation_matrix']

        with open(f'datasets/results/correlation_matrix_{ds_name}.csv', 'w') as f:
            # 写表头
            f.write(',' + ','.join(method_names) + '\n')
            # 写数据
            for i, m in enumerate(method_names):
                row = m + ',' + ','.join([f"{corr_matrix[i][j]:.4f}" for j in range(len(method_names))])
                f.write(row + '\n')

        print(f"相关系数矩阵CSV已保存: datasets/results/correlation_matrix_{ds_name}.csv")

    # 生成汇总表
    summary = []
    for ds_name, result in results.items():
        summary.append({
            'dataset': ds_name,
            'avg_correlation': result['avg_correlation'],
            'avg_std': result['avg_std']
        })

    with open('datasets/results/divergence_summary.csv', 'w') as f:
        f.write('Dataset,AvgCorrelation,AvgStd\n')
        for s in summary:
            f.write(f"{s['dataset']},{s['avg_correlation']:.4f},{s['avg_std']:.4f}\n")

    print(f"分歧度汇总表已保存: datasets/results/divergence_summary.csv")

if __name__ == '__main__':
    main()