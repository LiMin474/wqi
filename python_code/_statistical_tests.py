"""
统计显著性检验
使用Wilcoxon和Friedman检验证明集成提升显著
"""
import numpy as np
import json
from scipy.stats import wilcoxon, friedmanchisquare

def main():
    # 加载分歧度分析结果（包含各算法预测值）
    with open('datasets/results/divergence_analysis.json', 'r') as f:
        divergence_data = json.load(f)

    # 加载集成结果
    with open('datasets/results/new_ensemble_results.json', 'r') as f:
        ensemble_data = json.load(f)

    results = {}

    # 数据集名称映射
    ds_mapping = {
        'AKH': '4_akh_wqi',
        'WQI': '2_wqi_dataset',
        'Sample': '3_sample_dataset',
        'Jajpur': '1_jajpur'
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

        # 计算集成预测（WeightedAvg）
        # 从集成结果获取权重
        ensemble_key = ds_mapping.get(ds_name, ds_name)
        ensemble_result = ensemble_data.get(ensemble_key, {})
        weighted_avg_result = ensemble_result.get('WeightedAvg', {})
        weights = weighted_avg_result.get('Weights', {})

        if weights:
            # WeightedAvg集成预测
            ensemble_pred = np.zeros(len(y_test))
            for method in method_names:
                w = weights.get(method, 1.0 / n_methods)
                ensemble_pred += w * np.array(predictions[method])

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
                # 构建数据矩阵：每个样本在各算法上的误差
                error_matrix = []
                for method in method_names:
                    error_matrix.append(np.abs(y_test - np.array(predictions[method])))
                error_matrix.append(ensemble_errors)  # 加入集成

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
        else:
            print("  未找到权重信息，跳过集成检验")
            results[ds_name] = {
                'mae_values': mae_values,
                'ensemble_mae': None,
                'wilcoxon_p': None,
                'friedman_p': None
            }

    # 保存结果
    with open('datasets/results/statistical_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n统计检验结果已保存: datasets/results/statistical_tests.json")

    # 生成CSV汇总表
    with open('datasets/results/statistical_tests_summary.csv', 'w') as f:
        f.write('Dataset,BestSingle,SingleMAE,EnsembleMAE,Improvement(%),WilcoxonP,FriedmanP,Significant\n')
        for ds_name, r in results.items():
            if r['ensemble_mae']:
                sig = 'Yes' if r['wilcoxon_p'] < 0.05 else 'No'
                f.write(f"{ds_name},{r['best_single_method']},{r['best_single_mae']:.4f},{r['ensemble_mae']:.4f},{r['improvement']:.2f},{r['wilcoxon_p']:.4f},{r['friedman_p']:.4f},{sig}\n")

    print(f"统计检验汇总表已保存: datasets/results/statistical_tests_summary.csv")

if __name__ == '__main__':
    main()