"""
困难样本分析
分析哪些样本被多数模型预测错误
"""
import numpy as np
import json
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_APSM_jSO_fitrnet_opt import a4_APSM_jSO_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt

def analyze_difficult_samples(ds_name, ds_path, methods):
    """分析困难样本"""
    print(f"\n分析 {ds_name}...")

    # 加载数据
    data = np.load(ds_path)
    X, y = data['X'], data['y']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练所有模型
    predictions = {}
    for method_name, method_func in methods.items():
        print(f"  训练 {method_name}...", end=' ')
        Mdl, result = method_func(X_train, y_train)
        y_pred = Mdl.predict(X_test)
        predictions[method_name] = y_pred
        r2cv = result.get('R2CV', 0)
        print(f"R2CV={r2cv:.4f}")

    # 计算每个样本被多少模型预测错误
    errors_per_sample = np.zeros(len(y_test))
    r2_per_model = {}

    for method_name, y_pred in predictions.items():
        # 计算该模型的R2
        r2 = r2_score(y_test, y_pred)
        r2_per_model[method_name] = r2

        # 计算每个样本的误差
        sample_errors = np.abs(y_test - y_pred)
        threshold = np.mean(sample_errors) + 2 * np.std(sample_errors)

        # 标记错误样本
        is_wrong = sample_errors > threshold
        errors_per_sample += is_wrong.astype(int)

    # 困难样本分析
    majority_wrong = errors_per_sample >= len(methods) / 2
    difficult_indices = np.where(majority_wrong)[0]

    print(f"\n  困难样本统计:")
    print(f"    测试集样本数: {len(y_test)}")
    print(f"    被多数模型判错的样本数: {len(difficult_indices)} ({100*len(difficult_indices)/len(y_test):.1f}%)")

    if len(difficult_indices) > 0:
        difficult_wqi = y_test[difficult_indices]
        print(f"    困难样本WQI范围: {difficult_wqi.min():.1f} ~ {difficult_wqi.max():.1f}")
        print(f"    困难样本WQI均值: {difficult_wqi.mean():.1f}")

    # 计算集成预测
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    # 困难样本上的集成效果
    if len(difficult_indices) > 0:
        ensemble_errors = np.abs(y_test[difficult_indices] - ensemble_pred[difficult_indices])
        model_errors = []
        for method_name, y_pred in predictions.items():
            model_errors.append(np.mean(np.abs(y_test[difficult_indices] - y_pred[difficult_indices])))
        avg_model_error = np.mean(model_errors)

        print(f"    困难样本上平均模型MAE: {avg_model_error:.3f}")
        print(f"    困难样本上集成MAE: {np.mean(ensemble_errors):.3f}")
        improvement = (avg_model_error - np.mean(ensemble_errors)) / avg_model_error * 100
        print(f"    困难样本上集成改善: {improvement:.1f}%")

    return {
        'dataset': ds_name,
        'n_test': len(y_test),
        'n_difficult': len(difficult_indices),
        'difficult_ratio': len(difficult_indices) / len(y_test),
        'r2_per_model': r2_per_model,
        'ensemble_r2': ensemble_r2,
        'difficult_indices': difficult_indices.tolist(),
        'difficult_wqi': y_test[difficult_indices].tolist()
    }

def main():
    datasets = {
        'AKH': 'datasets/4_akh_wqi.npz',
        'WQI': 'datasets/2_wqi_dataset.npz'
    }

    methods = {
        'DE': a4_DE_fitrnet_opt,
        'SHADE': a4_SHADE_fitrnet_opt,
        'APSM-jSO': a4_APSM_jSO_fitrnet_opt,
        'CMA-ES': a4_CMAES_fitrnet_opt,
        'NRBO': a4_NRBO_fitrnet_opt
    }

    results = {}
    for ds_name, ds_path in datasets.items():
        results[ds_name] = analyze_difficult_samples(ds_name, ds_path, methods)

    # 保存结果
    with open('datasets/results/difficult_samples.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n困难样本分析结果已保存: datasets/results/difficult_samples.json")

    # 生成散点图数据（实际vs预测）
    for ds_name, result in results.items():
        ds_path = datasets[ds_name]
        data = np.load(ds_path)
        X, y = data['X'], data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型并预测
        predictions = {}
        for method_name, method_func in methods.items():
            Mdl, _ = method_func(X_train, y_train)
            predictions[method_name] = Mdl.predict(X_test)

        # 集成预测
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # 保存散点图数据
        scatter_data = []
        for i in range(len(y_test)):
            is_difficult = i in result['difficult_indices']
            scatter_data.append({
                'actual': float(y_test[i]),
                'ensemble_pred': float(ensemble_pred[i]),
                'is_difficult': is_difficult
            })

        with open(f'datasets/results/scatter_{ds_name}.json', 'w') as f:
            json.dump(scatter_data, f, indent=2)

        # 生成CSV
        with open(f'datasets/results/scatter_{ds_name}.csv', 'w') as f:
            f.write('Actual,Ensemble_Pred,IsDifficult\n')
            for item in scatter_data:
                f.write(f"{item['actual']:.4f},{item['ensemble_pred']:.4f},{item['is_difficult']}\n")

        print(f"散点图数据已保存: datasets/results/scatter_{ds_name}.csv")

if __name__ == '__main__':
    main()
