"""
WeightedAvg权重可视化
生成各数据集上5算法的权重分布数据
"""
import json
import numpy as np

def main():
    # 加载集成结果
    with open('datasets/results/new_ensemble_results.json', 'r') as f:
        ensemble_data = json.load(f)

    # 数据集名称映射
    ds_names = {
        '1_jajpur': 'Jajpur',
        '2_wqi_dataset': 'WQI',
        '3_sample_dataset': 'Sample',
        '4_akh_wqi': 'AKH'
    }

    methods = ['DE', 'SHADE', 'APSM-jSO', 'CMA-ES', 'NRBO']

    # 生成权重汇总表
    weights_summary = []

    for ds_key, ds_name in ds_names.items():
        if ds_key in ensemble_data:
            weighted_avg = ensemble_data[ds_key].get('WeightedAvg', {})
            weights = weighted_avg.get('Weights', {})

            if weights:
                print(f"\n{ds_name} 数据集权重分布:")
                for method in methods:
                    w = weights.get(method, 0)
                    print(f"  {method}: {w:.4f}")
                    weights_summary.append({
                        'Dataset': ds_name,
                        'Method': method,
                        'Weight': w
                    })

    # 保存CSV格式（用于Origin画柱状图）
    with open('datasets/results/weights_distribution.csv', 'w') as f:
        f.write('Dataset,Method,Weight\n')
        for w in weights_summary:
            f.write(f"{w['Dataset']},{w['Method']},{w['Weight']:.6f}\n")

    print(f"\n权重分布CSV已保存: datasets/results/weights_distribution.csv")

    # 生成矩阵格式（用于Origin画热图）
    with open('datasets/results/weights_matrix.csv', 'w') as f:
        f.write(',' + ','.join(methods) + '\n')
        for ds_key, ds_name in ds_names.items():
            if ds_key in ensemble_data:
                weighted_avg = ensemble_data[ds_key].get('WeightedAvg', {})
                weights = weighted_avg.get('Weights', {})
                row = ds_name
                for method in methods:
                    w = weights.get(method, 0)
                    row += f",{w:.4f}"
                f.write(row + '\n')

    print(f"权重矩阵CSV已保存: datasets/results/weights_matrix.csv")

    # 生成JSON汇总
    weights_json = {}
    for ds_key, ds_name in ds_names.items():
        if ds_key in ensemble_data:
            weighted_avg = ensemble_data[ds_key].get('WeightedAvg', {})
            weights = weighted_avg.get('Weights', {})
            weights_json[ds_name] = weights

    with open('datasets/results/weights_summary.json', 'w') as f:
        json.dump(weights_json, f, indent=2)

    print(f"权重汇总JSON已保存: datasets/results/weights_summary.json")

if __name__ == '__main__':
    main()