import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import mealpy
from mealpy.utils.space import IntegerVar, FloatVar

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))


def load_dataset(name):
    """Load .npz dataset from python_code/datasets/"""
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    target_name = str(data['target_name'])
    dataset_name = str(data['name'])
    return X, y, dataset_name, target_name


def objective_function(params, X, y, cv_folds=5):
    """
    目标函数：使用给定的超参数训练ANN并返回交叉验证R²
    params: numpy数组 [num_layers, layer_1, layer_2, activation, alpha]
    """
    try:
        # 解码参数 - params是数组格式
        num_layers = int(np.clip(params[0], 1, 2))  # 确保在1-2范围内
        layer_1 = max(1, int(params[1]))  # 确保至少1个神经元
        layer_2 = max(1, int(params[2])) if num_layers == 2 else 0  # 确保至少1个神经元
        activation_idx = int(np.clip(params[3], 0, 2))  # 确保在0-2范围内
        alpha = max(1e-4, 10 ** params[4])  # 确保alpha大于0

        # 激活函数映射
        activations = ['relu', 'tanh', 'logistic']
        activation = activations[activation_idx]

        # 构建网络结构
        if num_layers == 1:
            hidden_layer_sizes = (layer_1,)
        else:
            hidden_layer_sizes = (layer_1, layer_2)

        # 创建并训练模型
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 交叉验证
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
        r2cv = np.mean(cv_scores)

        return -r2cv  # mealpy是最小化，所以返回负R²

    except Exception as e:
        print(f'Objective function error: {e}')
        return 1.0  # 返回最差值


def run_mealpy_algorithm(algorithm_class, algorithm_name, X, y, max_generations=5, pop_size=5):
    """
    使用mealpy算法优化ANN超参数
    """
    # 定义搜索空间（使用mealpy 3.x的变量类型）
    bounds = [
        IntegerVar(lb=1, ub=2, name="num_layers"),     # 1或2层
        IntegerVar(lb=1, ub=50, name="layer_1"),      # 第一层神经元数
        IntegerVar(lb=1, ub=50, name="layer_2"),      # 第二层神经元数
        IntegerVar(lb=0, ub=2, name="activation"),    # 激活函数索引
        FloatVar(lb=-4, ub=0, name="alpha"),          # log10(alpha)
    ]

    # 创建问题对象
    problem = mealpy.Problem(bounds=bounds, minmax="min", obj_func=lambda params: objective_function(params, X, y))

    # 初始化算法
    optimizer = algorithm_class(epoch=max_generations, pop_size=pop_size)

    # 运行优化 - mealpy 3.x 返回 Agent 对象
    t0 = time.time()
    best_agent = optimizer.solve(problem)
    elapsed = time.time() - t0

    # 获取最优参数和适应度
    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    # 解码最优参数
    num_layers = int(best_position[0])
    layer_1 = int(best_position[1])
    layer_2 = int(best_position[2]) if num_layers == 2 else 0
    activation_idx = int(best_position[3])
    alpha = 10 ** best_position[4]

    activations = ['relu', 'tanh', 'logistic']
    activation = activations[activation_idx]

    # 使用最优参数训练最终模型
    if num_layers == 1:
        hidden_layer_sizes = (layer_1,)
    else:
        hidden_layer_sizes = (layer_1, layer_2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    model.fit(X_scaled, y)

    # 计算所有指标
    y_pred = model.predict(X_scaled)
    r2 = model.score(X_scaled, y)
    r2cv = -best_fitness  # 转换回正数
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    return {
        'R2': r2,
        'R2CV': r2cv,
        'RMSE': rmse,
        'MAE': mae,
        'NumLayers': num_layers,
        'Layer_1': layer_1,
        'Layer_2': layer_2,
        'Activation': activation,
        'Alpha': alpha,
        'Time': elapsed
    }


def run_experiment(dataset_key, X, y, dataset_name, target_name):
    print()
    print('#' * 70)
    print(f'# Dataset: {dataset_name}')
    print(f'# Samples: {len(y)}, Features: {X.shape[1]}, Target: {target_name}')
    print(f'# Target stats: mean={y.mean():.3f}, std={y.std():.3f}')
    print('#' * 70)

    results = {}

    # 定义要测试的新算法（跳过RUN，因其运行时间过长）
    algorithms = {
        'RIME (2023)': mealpy.physics_based.RIME.OriginalRIME,
        'INFO (2022)': mealpy.math_based.INFO.OriginalINFO,
        'GJO (2022)': mealpy.swarm_based.GJO.OriginalGJO,
    }

    for algorithm_name, algorithm_class in algorithms.items():
        print()
        print(f'--- Running {algorithm_name} ---')
        try:
            result = run_mealpy_algorithm(algorithm_class, algorithm_name, X, y)
            results[algorithm_name] = result
            print(f'  {algorithm_name}:')
            print(f'    R2={result["R2"]:.4f}, R2CV={result["R2CV"]:.4f}, '
                  f'RMSE={result["RMSE"]:.4f}, MAE={result["MAE"]:.4f}, '
                  f'Time={result["Time"]:.1f}s')
            layer2_str = f', {result["Layer_2"]}' if result["NumLayers"] == 2 else ''
            print(f'    Architecture: {result["NumLayers"]} layer(s), '
                  f'[{result["Layer_1"]}{layer2_str}], '
                  f'{result["Activation"]}, alpha={result["Alpha"]:.4f}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  {algorithm_name}: FAILED - {str(e)[:200]}')
            results[algorithm_name] = {'R2': np.nan, 'R2CV': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Error': str(e)[:200]}

    return results


def print_comparison(dataset_key, results):
    print()
    print(f'{"=" * 90}')
    print(f'  Results Summary: {dataset_key}')
    print(f'{"=" * 90}')
    print(f'{"Method":<15} {"R2":>10} {"R2CV":>10} {"RMSE":>10} {"MAE":>10} {"Layers":>8} {"L1":>5} {"L2":>5} {"Act":>8} {"Alpha":>12} {"Time(s)":>8}')
    print(f'{"-" * 90}')

    for method_name, result in results.items():
        if 'Error' in result:
            print(f'{method_name:<15} {"ERROR":>10} {"ERROR":>10} {"ERROR":>10} {"ERROR":>10} {"ERROR":>8} {"ERROR":>5} {"ERROR":>5} {"ERROR":>8} {"ERROR":>12} {"ERROR":>8}')
        else:
            print(f'{method_name:<15} {result["R2"]:>10.4f} {result["R2CV"]:>10.4f} '
                  f'{result["RMSE"]:>10.4f} {result["MAE"]:>10.4f} '
                  f'{result["NumLayers"]:>8} {result["Layer_1"]:>5} {result["Layer_2"]:>5} '
                  f'{result["Activation"][:8]:>8} {result["Alpha"]:>12.4f} {result["Time"]:>8.1f}')

    print(f'{"=" * 90}')


def generate_md_report(all_results):
    """生成Markdown格式的实验报告"""
    md_content = "# 新进化算法实验结果\n\n"
    md_content += "## 实验概述\n\n"
    md_content += "本实验使用以下3种新进化算法对ANN超参数进行优化：\n"
    md_content += "- **RIME (2023)**: 霜冰优化算法\n"
    md_content += "- **INFO (2022)**: 信息算法\n"
    md_content += "- **GJO (2022)**: 金豺优化算法\n\n"
    md_content += "测试数据集：Jajpur、WQI、Sample、AKH\n\n"
    md_content += "## 实验结果\n\n"

    for dataset_name, results in all_results.items():
        md_content += f"### {dataset_name} 数据集\n\n"
        md_content += "| 算法 | R² | R²CV | RMSE | MAE | 层数 | L1 | L2 | 激活函数 | Alpha | 时间(s) |\n"
        md_content += "|:-----|:----:|:-----:|:----:|:---:|:----:|:---:|:---:|:--------:|:------:|:--------:|\n"
        
        for method_name, result in results.items():
            if 'Error' in result:
                md_content += f"| {method_name} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n"
            else:
                md_content += f"| {method_name} | {result['R2']:.4f} | {result['R2CV']:.4f} | {result['RMSE']:.4f} | {result['MAE']:.4f} | "
                md_content += f"{result['NumLayers']} | {result['Layer_1']} | {result['Layer_2']} | {result['Activation']} | "
                md_content += f"{result['Alpha']:.4f} | {result['Time']:.1f} |\n"
        
        md_content += "\n"

    md_content += "## 结果汇总\n\n"
    md_content += "### 各数据集最优R²CV\n\n"
    md_content += "| 数据集 | RIME (2023) | INFO (2022) | GJO (2022) | 最优算法 |\n"
    md_content += "|:-------|:-----------:|:-----------:|:----------:|:--------:|\n"
    
    for dataset_name, results in all_results.items():
        best_alg = ""
        best_r2cv = -1
        row = f"| {dataset_name} | "
        for method_name, result in results.items():
            if 'Error' not in result:
                r2cv = result['R2CV']
                row += f"{r2cv:.4f} | "
                if r2cv > best_r2cv:
                    best_r2cv = r2cv
                    best_alg = method_name.split()[0]
            else:
                row += "ERROR | "
        row += f"| {best_alg} |\n"
        md_content += row

    return md_content


def main():
    datasets = {
        '1_jajpur': 'Jajpur',
        '2_wqi_dataset': 'WQI',
        '3_sample_dataset': 'Sample',
        '4_akh_wqi': 'AKH'
    }

    all_results = {}

    for dataset_key, dataset_name in datasets.items():
        try:
            X, y, loaded_name, target_name = load_dataset(dataset_key)
            results = run_experiment(dataset_key, X, y, dataset_name, target_name)
            all_results[dataset_name] = results
            print_comparison(dataset_name, results)

            # 保存单个数据集结果
            output_file = os.path.join(SCRIPT_DIR, 'datasets', 'results', f'{dataset_name}_new_algorithms.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'\nResults saved to: {output_file}')

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'\nERROR processing {dataset_key}: {str(e)}')
            all_results[dataset_name] = {'Error': str(e)}

    # 保存所有结果
    output_file = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_new_algorithms_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n\nAll results saved to: {output_file}')

    # 生成Markdown报告
    md_report = generate_md_report(all_results)
    md_file = os.path.join(SCRIPT_DIR, '..', '新进化算法.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f'Markdown report saved to: {md_file}')

    return all_results


if __name__ == '__main__':
    results = main()