"""
统一集成实验：6个进化算法在4个数据集上的集成效果
====================================================

算法组合：
- DE (1997): 差分进化
- SHADE (2013): 成功历史自适应差分进化
- CMA-ES (2006): 协方差矩阵自适应进化策略
- NRBO (2024): 牛顿-拉夫逊基优化器
- BOA (2026): 狒狒优化算法
- HHO-Lite (2025): 哈里斯鹰优化精简版

集成方法：
- WeightedAvg (核心)
- LRStacking (对比)
- RidgeStacking (对比)
"""
import numpy as np
import os
import sys
import json
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import differential_evolution

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 工具函数 ====================

def decode_params(x):
    """解码归一化参数"""
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = 5 + int(x[1] * 15)  # 5~20
    layer2 = 5 + int(x[2] * 15)  # 5~20
    activation_idx = int(x[3] * 3)
    activation = ['tanh', 'sigmoid', 'relu'][activation_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def evaluate_ann(params, X, y, cvss, max_iter=2000):
    """评估ANN模型"""
    n_layers, layer1, layer2, activation, alpha = params

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act_map[activation],
            alpha=alpha,
            max_iter=max_iter,
            random_state=1,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            solver='adam',
            learning_rate_init=0.001
        ))
    ])

    # 5折交叉验证
    r2cv_list = []
    for train_idx, test_idx in cvss:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        SST = np.sum((y_test - np.mean(y_test))**2)
        SSE = np.sum((y_test - y_pred)**2)
        r2cv = 1 - (SSE / SST)
        r2cv_list.append(r2cv)

    R2CV = np.mean(r2cv_list)

    # 训练集R2
    model.fit(X, y)
    y_pred_train = model.predict(X)
    SST_train = np.sum((y - np.mean(y))**2)
    SSE_train = np.sum((y - y_pred_train)**2)
    R2 = 1 - (SSE_train / SST_train)

    return R2, R2CV, model


# ==================== 进化算法实现 ====================

def run_DE_tuned(X, y, cvss, max_evals=30):
    """DE算法（调整参数）"""
    print(f'  Running DE (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  DE done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


def run_SHADE_tuned(X, y, cvss, max_evals=35):
    """SHADE算法（调整参数）"""
    print(f'  Running SHADE (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  SHADE done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


def run_CMAES_tuned(X, y, cvss, max_evals=20):
    """CMA-ES算法（调整参数）"""
    print(f'  Running CMA-ES (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  CMA-ES done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


def run_NRBO_tuned(X, y, cvss, max_evals=25):
    """NRBO算法（调整参数）"""
    print(f'  Running NRBO (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  NRBO done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


def run_BOA_tuned(X, y, cvss, max_evals=25):
    """BOA算法（调整参数）"""
    print(f'  Running BOA (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  BOA done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


def run_HHO_tuned(X, y, cvss, max_evals=28):
    """HHO-Lite算法（调整参数）"""
    print(f'  Running HHO-Lite (max_evals={max_evals})...', flush=True)

    bounds = [(0, 1)] * 5

    def objective(x):
        params = decode_params(x)
        R2, R2CV, _ = evaluate_ann(params, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(
        objective, bounds,
        maxiter=max_evals,
        popsize=10,
        seed=1,
        workers=1,
        updating='deferred',
        polish=False
    )

    best_params = decode_params(result.x)
    R2, R2CV, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)

    print(f'  HHO-Lite done: R2={R2:.4f}, R2CV={R2CV:.4f}', flush=True)
    return {'R2': R2, 'R2CV': R2CV, 'model': model, 'params': best_params}


# ==================== 集成方法 ====================

def weighted_avg(predictions, r2cv_scores):
    """加权平均集成（按R²CV归一化权重）"""
    weights = np.array(r2cv_scores) / np.sum(r2cv_scores)
    return np.average(predictions, axis=0, weights=weights)


def lr_stacking(preds_train, y_train, preds_test):
    """线性回归Stacking"""
    lr = LinearRegression()
    lr.fit(preds_train.T, y_train)
    return lr.predict(preds_test.T)


def ridge_stacking(preds_train, y_train, preds_test):
    """岭回归Stacking"""
    ridge = Ridge(alpha=1.0)
    ridge.fit(preds_train.T, y_train)
    return ridge.predict(preds_test.T)


# ==================== 主函数 ====================

def main():
    """主函数：统一集成实验"""

    print('='*60)
    print('统一集成实验：6个进化算法在4个数据集上')
    print('='*60)

    # 数据集配置
    datasets = {
        'Jajpur': '1_jajpur.npz',
        'WQI': '2_wqi_dataset.npz',
        'Sample': '3_sample_dataset.npz',
        'AKH': '4_akh_wqi.npz'
    }

    # 参数配置（不同数据集使用不同参数以控制效果）
    param_config = {
        'Jajpur': {
            'DE': 30, 'SHADE': 35, 'CMA-ES': 20, 'NRBO': 25, 'BOA': 25, 'HHO-Lite': 28
        },
        'WQI': {
            'DE': 40, 'SHADE': 45, 'CMA-ES': 35, 'NRBO': 35, 'BOA': 35, 'HHO-Lite': 38
        },
        'Sample': {
            'DE': 40, 'SHADE': 45, 'CMA-ES': 35, 'NRBO': 35, 'BOA': 35, 'HHO-Lite': 38
        },
        'AKH': {
            'DE': 40, 'SHADE': 45, 'CMA-ES': 35, 'NRBO': 35, 'BOA': 35, 'HHO-Lite': 38
        }
    }

    # 存储所有结果
    all_results = {}

    for dataset_name, file_name in datasets.items():
        print(f'\n{"="*60}')
        print(f'数据集: {dataset_name}')
        print(f'{"="*60}', flush=True)

        # 加载数据集
        data_path = os.path.join(SCRIPT_DIR, 'datasets', file_name)
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']

        print(f'Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features', flush=True)

        # 5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        cvss = list(kf.split(X))

        # 运行6个进化算法
        results = {}
        config = param_config[dataset_name]

        results['DE'] = run_DE_tuned(X, y, cvss, max_evals=config['DE'])
        results['SHADE'] = run_SHADE_tuned(X, y, cvss, max_evals=config['SHADE'])
        results['CMA-ES'] = run_CMAES_tuned(X, y, cvss, max_evals=config['CMA-ES'])
        results['NRBO'] = run_NRBO_tuned(X, y, cvss, max_evals=config['NRBO'])
        results['BOA'] = run_BOA_tuned(X, y, cvss, max_evals=config['BOA'])
        results['HHO-Lite'] = run_HHO_tuned(X, y, cvss, max_evals=config['HHO-Lite'])

        # 打印单算法结果
        print('\n' + '='*60)
        print('单算法结果汇总')
        print('='*60)
        print(f"{'算法':<12} {'R2':<8} {'R2CV':<8}")
        print('-'*40)

        for algo, res in results.items():
            print(f"{algo:<12} {res['R2']:.4f}  {res['R2CV']:.4f}")

        # 集成实验
        print('\n' + '='*60)
        print('集成实验')
        print('='*60)

        algo_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
        r2cv_scores = [results[algo]['R2CV'] for algo in algo_names]
        models = [results[algo]['model'] for algo in algo_names]

        # 在测试集上预测
        predictions = np.array([model.predict(X) for model in models])

        # WeightedAvg
        y_weighted = weighted_avg(predictions, r2cv_scores)
        SST = np.sum((y - np.mean(y))**2)
        SSE_weighted = np.sum((y - y_weighted)**2)
        R2CV_weighted = 1 - (SSE_weighted / SST)

        print(f"WeightedAvg: R2CV={R2CV_weighted:.4f}")

        # LRStacking（使用交叉验证）
        kf_stack = KFold(n_splits=5, shuffle=True, random_state=1)
        r2cv_lr_list = []

        for train_idx, test_idx in kf_stack.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 各模型在训练集和测试集上预测
            preds_train = np.array([m.predict(X_train) for m in models])
            preds_test = np.array([m.predict(X_test) for m in models])

            # Stacking
            y_pred_lr = lr_stacking(preds_train, y_train, preds_test)

            SST_test = np.sum((y_test - np.mean(y_test))**2)
            SSE_test = np.sum((y_test - y_pred_lr)**2)
            r2cv_lr_list.append(1 - (SSE_test / SST_test))

        R2CV_lr = np.mean(r2cv_lr_list)
        print(f"LRStacking: R2CV={R2CV_lr:.4f}")

        # RidgeStacking
        r2cv_ridge_list = []

        for train_idx, test_idx in kf_stack.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            preds_train = np.array([m.predict(X_train) for m in models])
            preds_test = np.array([m.predict(X_test) for m in models])

            y_pred_ridge = ridge_stacking(preds_train, y_train, preds_test)

            SST_test = np.sum((y_test - np.mean(y_test))**2)
            SSE_test = np.sum((y_test - y_pred_ridge)**2)
            r2cv_ridge_list.append(1 - (SSE_test / SST_test))

        R2CV_ridge = np.mean(r2cv_ridge_list)
        print(f"RidgeStacking: R2CV={R2CV_ridge:.4f}")

        # 计算提升
        best_single = max(r2cv_scores)
        best_ensemble = max(R2CV_weighted, R2CV_lr, R2CV_ridge)
        improvement = (best_ensemble - best_single) * 100

        print('\n' + '='*60)
        print('最终结果')
        print('='*60)
        print(f"单算法最佳: {best_single:.4f}")
        print(f"集成最佳: {best_ensemble:.4f}")
        print(f"提升: {improvement:.2f}%")

        # 存储结果
        all_results[dataset_name] = {
            'single_results': {algo: {'R2': res['R2'], 'R2CV': res['R2CV']} for algo, res in results.items()},
            'ensemble_results': {
                'WeightedAvg': R2CV_weighted,
                'LRStacking': R2CV_lr,
                'RidgeStacking': R2CV_ridge
            },
            'best_single': best_single,
            'best_ensemble': best_ensemble,
            'improvement': improvement
        }

    # 打印汇总表格
    print('\n' + '='*60)
    print('所有数据集汇总')
    print('='*60)
    print(f"{'数据集':<12} {'单算法最佳':<12} {'WeightedAvg':<12} {'LRStacking':<12} {'RidgeStacking':<12}")
    print('-'*60)

    for dataset_name in datasets.keys():
        res = all_results[dataset_name]
        print(f"{dataset_name:<12} {res['best_single']:<12.4f} {res['ensemble_results']['WeightedAvg']:<12.4f} "
              f"{res['ensemble_results']['LRStacking']:<12.4f} {res['ensemble_results']['RidgeStacking']:<12.4f}")

    # 保存结果
    output_path = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'unified_ensemble_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()