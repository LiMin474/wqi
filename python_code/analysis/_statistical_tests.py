"""
统计显著性检验
使用Wilcoxon和Friedman检验证明集成提升显著
"""
import numpy as np
import json, os
from scipy.stats import wilcoxon, friedmanchisquare

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # python_code/
    RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

    # 加载分歧度分析结果（包含各算法预测值）
    div_path = os.path.join(RESULTS_DIR, 'divergence_analysis.json')
    if not os.path.exists(div_path):
        print(f"请先运行 _analyze_divergence.py 生成 {div_path}")
        return
    with open(div_path, 'r') as f:
        divergence_data = json.load(f)

    # 加载集成结果
    ens_path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
    if not os.path.exists(ens_path):
        print(f"请先运行 _run_unified_ensemble.py 生成 {ens_path}")
        return
    with open(ens_path, 'r') as f:
        ensemble_data = json.load(f)

    results = {}

    # 数据集名称映射
    ds_mapping = {
        'AKH': 'AKH',
        'Irish': 'Irish',
        'Jajpur': 'Jajpur'
    }

    for ds_name in divergence_data.keys():
        print(f"\n=== {ds_name} 数据集统计检验 ===")

        # 获取各算法预测值
        predictions = divergence_data[ds_name]['predictions']
        y_test = np.array(divergence_data[ds_name]['y_test'])

        method_names = list(predictions.keys())
        n_methods = len(method_names)

        # 计算各算法的MAE
        mae_values = {}
        for method in method_names:
            y_pred = np.array(predictions[method])
            mae = np.mean(np.abs(y_test - y_pred))
            mae_values[method] = mae

        print(f"各算法MAE:")
        for method, mae in mae_values.items():
            print(f"  {method}: {mae:.4f}")

        # 计算集成预测（R²CV加权平均）
        ensemble_key = ds_mapping.get(ds_name, ds_name)
        ens_result = ensemble_data.get(ensemble_key, {})
        single_results = ens_result.get('single_results', {})

        # 计算权重
        r2cv_values = []
        for method in method_names:
            r2cv = single_results.get(method, {}).get('R2CV', 0)
            r2cv_values.append(max(r2cv, 0))  # 防止负权重

        if sum(r2cv_values) > 0:
            weights = np.array(r2cv_values) / np.sum(r2cv_values)
        else:
            weights = np.ones(n_methods) / n_methods

        # WeightedAvg集成预测
        ensemble_pred = np.zeros(len(y_test))
        for i, method in enumerate(method_names):
            ensemble_pred += weights[i] * np.array(predictions[method])

        ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
        print(f"  WeightedAvg集成: {ensemble_mae:.4f}")

        # Wilcoxon检验：集成 vs 最优单算法
        best_single_method = min(mae_values, key=mae_values.get)
        best_single_mae = mae_values[best_single_method]

        # 计算逐样本误差
        ensemble_errors = np.abs(y_test - ensemble_pred)
        best_single_errors = np.abs(y_test - np.array(predictions[best_single_method]))

        # Wilcoxon符号秩检验
        try:
            stat, p_value = wilcoxon(ensemble_errors, best_single_errors)
            print(f"\nWilcoxon检验 (集成 vs {best_single_method}):")
            print(f"  统计量: {stat:.4f}")
            print(f"  p值: {p_value:.4f}")
            print(f"  结论: {'显著' if p_value < 0.05 else '不显著'} (p<0.05)")
        except Exception as e:
            print(f"  Wilcoxon检验失败: {e}")
            p_value = 1.0

        # Friedman检验：所有算法比较
        try:
            error_matrix = []
            for method in method_names:
                error_matrix.append(np.abs(y_test - np.array(predictions[method])))
            error_matrix.append(ensemble_errors)

            stat, p_value_friedman = friedmanchisquare(*error_matrix)
            print(f"\nFriedman检验 (所有算法+集成):")
            print(f"  统计量: {stat:.4f}")
            print(f"  p值: {p_value_friedman:.4f}")
            print(f"  结论: {'显著差异' if p_value_friedman < 0.05 else '无显著差异'} (p<0.05)")
        except Exception as e:
            print(f"  Friedman检验失败: {e}")
            p_value_friedman = 1.0

        results[ds_name] = {
            'mae_values': mae_values,
            'ensemble_mae': ensemble_mae,
            'best_single_method': best_single_method,
            'best_single_mae': best_single_mae,
            'improvement': (best_single_mae - ensemble_mae) / best_single_mae * 100,
            'wilcoxon_p': p_value,
            'friedman_p': p_value_friedman
        }

    # 保存结果
    out_path = os.path.join(RESULTS_DIR, 'statistical_tests.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n统计检验结果已保存: {out_path}")

    # 生成CSV汇总表
    csv_path = os.path.join(RESULTS_DIR, 'statistical_tests_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Dataset,BestSingle,SingleMAE,EnsembleMAE,Improvement(%),WilcoxonP,FriedmanP,Significant\n')
        for ds_name, r in results.items():
            if r['ensemble_mae'] is not None:
                sig = 'Yes' if r['wilcoxon_p'] < 0.05 else 'No'
                f.write(f"{ds_name},{r['best_single_method']},{r['best_single_mae']:.4f},{r['ensemble_mae']:.4f},{r['improvement']:.2f},{r['wilcoxon_p']:.4f},{r['friedman_p']:.4f},{sig}\n")

    print(f"统计检验汇总表已保存: {csv_path}")

if __name__ == '__main__':
    main()