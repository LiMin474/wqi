"""
6种进化算法集成对比实验
===================================
对每个数据集，对比4种集成策略：
  1. SimpleAvg      - 简单平均
  2. WeightedR2CV   - 按R2CV分配权重
  3. LRStacking     - LinearRegression元学习器
  4. RidgeStacking  - Ridge回归元学习器

基学习器：DE, SHADE, CMA-ES, NRBO, BOA, HHO-Lite（6个进化算法）

使用快速模式：从已有JSON读取超参数，重建ANN，避免重新优化。
"""

import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

# 导入六个进化算法
from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt
from common_codes.a4_BOA_fitrnet_opt import a4_BOA_fitrnet_opt
from common_codes.a4_HHO_Lite_fitrnet_opt import a4_HHO_Lite_fitrnet_opt


# 六个进化算法
ALGO_NAMES = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
ALGO_FUNCS = [a4_DE_fitrnet_opt, a4_SHADE_fitrnet_opt,
              a4_CMAES_fitrnet_opt, a4_NRBO_fitrnet_opt,
              a4_BOA_fitrnet_opt, a4_HHO_Lite_fitrnet_opt]

N_FOLDS = 5
RANDOM_STATE = 1


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    return data['X'], data['y'], str(data['name']), str(data['target_name'])


def build_ann(params):
    """根据超参数构建ANN模型"""
    n_layers = params['NumLayers']
    layer1 = params['Layer_1']
    layer2 = params['Layer_2']
    activation = params['Activation']
    alpha = params.get('Alpha', 1e-6)
    if np.isnan(alpha) or alpha is None:
        alpha = 1e-6

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
    act = act_map.get(activation, 'relu')

    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act,
            solver='lbfgs',
            alpha=alpha,
            max_iter=2000,
            random_state=RANDOM_STATE,
            early_stopping=True
        ))
    ])


def get_oof_predictions(X, y, algo_func, algo_name, n_folds=N_FOLDS):
    """对单个算法生成5折out-of-fold预测"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(n)
    fold_r2_list = []

    for train_idx, val_idx in kf.split(X):
        Mdl, _ = algo_func(X[train_idx], y[train_idx])
        oof_preds[val_idx] = Mdl.predict(X[val_idx])
        sst = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
        ssr = np.sum((y[val_idx] - oof_preds[val_idx]) ** 2)
        r2 = 1 - ssr / sst if sst > 0 else 0
        fold_r2_list.append(r2)

    return oof_preds, fold_r2_list


def ensemble_simple_avg(oof_preds_matrix, y):
    """方法1：简单平均"""
    preds = np.mean(oof_preds_matrix, axis=1)
    return preds


def ensemble_weighted_r2cv(oof_preds_matrix, y, algo_r2cv):
    """方法2：按R2CV加权平均"""
    weights = np.array([algo_r2cv[algo] for algo in ALGO_NAMES])
    weights = weights / weights.sum()
    preds = np.average(oof_preds_matrix, axis=1, weights=weights)
    return preds


def ensemble_weighted_r2cv2(oof_preds_matrix, y, algo_r2cv):
    """方法3：按R2CV平方加权（放大差异）"""
    weights = np.array([algo_r2cv[algo] ** 2 for algo in ALGO_NAMES])
    weights = weights / weights.sum()
    preds = np.average(oof_preds_matrix, axis=1, weights=weights)
    return preds


def ensemble_lr_stacking(oof_preds_matrix, y):
    """方法4：LR Stacking（元学习器LinearRegression）"""
    meta = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    meta.fit(oof_preds_matrix, y)
    preds = meta.predict(oof_preds_matrix)
    return preds


def ensemble_ridge_stacking(oof_preds_matrix, y):
    """方法4b：Ridge Stacking（元学习器Ridge回归，防止过拟合）"""
    meta = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    meta.fit(oof_preds_matrix, y)
    preds = meta.predict(oof_preds_matrix)
    return preds


def evaluate(preds, y, algo_r2cv=None):
    """计算R2/R2CV/RMSE/MAE"""
    r2 = r2_score(y, preds)

    # R2CV: 5折CV评估
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_r2_list = []
    for train_idx, val_idx in kf.split(preds):
        sst = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
        ssr = np.sum((y[val_idx] - preds[val_idx]) ** 2)
        fold_r2_list.append(1 - ssr / sst if sst > 0 else 0)
    r2cv = float(np.mean(fold_r2_list))

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, r2cv, rmse, mae


def run_ensemble_for_dataset(X, y, dataset_name):
    """对单个数据集跑5种集成方法"""
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'Dataset: {dataset_name}  (n={len(y)})')
    print(f'{sep}')

    # Step 1: 对每个算法生成 out-of-fold 预测
    print(f'\n--- Step 1: Generating out-of-fold predictions (5-fold CV) ---')
    oof_preds_matrix = np.zeros((len(y), len(ALGO_NAMES)))
    algo_r2cv = {}

    for j, (name, func) in enumerate(zip(ALGO_NAMES, ALGO_FUNCS)):
        t0 = time.time()
        oof, fold_r2_list = get_oof_predictions(X, y, func, name)
        elapsed = time.time() - t0
        oof_preds_matrix[:, j] = oof
        r2cv = float(np.mean(fold_r2_list))
        algo_r2cv[name] = r2cv
        print(f'  {name:<10s}: R2CV={r2cv:.4f}  (fold R2: {[f"{r:.4f}" for r in fold_r2_list]})  [{elapsed:.1f}s]')

    # 打印基学习器权重
    print(f'\n  R2CV权重: { {k: f"{v:.4f}" for k, v in algo_r2cv.items()} }')
    w1 = np.array(list(algo_r2cv.values()))
    w1 = w1 / w1.sum()
    w2 = np.array(list(algo_r2cv.values())) ** 2
    w2 = w2 / w2.sum()
    print(f'  平均权重: {[f"{x:.3f}" for x in w1]}')
    print(f'  R2CV2权重: {[f"{x:.3f}" for x in w2]}')

    # Step 2: 对比5种集成方法
    print(f'\n--- Step 2: Ensemble Methods Comparison ---')
    results = {}

    # 方法1: 简单平均
    preds1 = ensemble_simple_avg(oof_preds_matrix, y)
    r2, r2cv, rmse, mae = evaluate(preds1, y)
    results['SimpleAvg'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [1] SimpleAvg     R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法2: 加权R2CV
    preds2 = ensemble_weighted_r2cv(oof_preds_matrix, y, algo_r2cv)
    r2, r2cv, rmse, mae = evaluate(preds2, y)
    results['WeightedR2CV'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [2] WeightedR2CV  R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法3: 加权R2CV^2
    preds3 = ensemble_weighted_r2cv2(oof_preds_matrix, y, algo_r2cv)
    r2, r2cv, rmse, mae = evaluate(preds3, y)
    results['WeightedR2CV2'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [3] WeightedR2CV2 R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法4: LR Stacking
    preds4 = ensemble_lr_stacking(oof_preds_matrix, y)
    r2, r2cv, rmse, mae = evaluate(preds4, y)
    results['LRStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [4] LRStacking   R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法4b: Ridge Stacking
    preds5 = ensemble_ridge_stacking(oof_preds_matrix, y)
    r2, r2cv, rmse, mae = evaluate(preds5, y)
    results['RidgeStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [5] RidgeStacking R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 找出最优方法
    best_method = max(results, key=lambda k: results[k]['R2CV'])
    best_r2cv = results[best_method]['R2CV']
    print(f'\n  >>> Best: {best_method} (R2CV={best_r2cv:.4f})')

    # 同时输出最优单算法
    best_single = max(algo_r2cv, key=lambda k: algo_r2cv[k])
    best_single_r2cv = algo_r2cv[best_single]
    improvement = best_r2cv - best_single_r2cv
    print(f'  >>> Best Single: {best_single} (R2CV={best_single_r2cv:.4f})')
    print(f'  >>> Ensemble Gain: +{improvement:.4f} over {best_single}')

    return results, algo_r2cv, best_method


def main():
    datasets = {
        '1_jajpur': '1_jajpur',
        '2_wqi_dataset': '2_wqi_dataset',
        '3_sample_dataset': '3_sample_dataset',
        '4_akh_wqi': '4_akh_wqi',
    }

    all_results = {}
    all_single_algo = {}

    for key, filename in datasets.items():
        X, y, dataset_name, target_name = load_dataset(filename)
        sep = '#' * 80
        print(f'\n\n{sep}')
        print(f"# Dataset {key}: {dataset_name} | n={len(y)}, features={X.shape[1]}, target={target_name}")
        print(f"# Target stats: mean={y.mean():.3f}, std={y.std():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")
        print(f'# {sep}')

        results, algo_r2cv, best_method = run_ensemble_for_dataset(X, y, dataset_name)
        all_results[key] = results
        all_single_algo[key] = algo_r2cv

    # ===== 全局汇总 =====
    sep = '#' * 80
    print(f'\n\n{sep}')
    print(f"# FINAL SUMMARY: 5 Ensemble Methods x 4 Datasets")
    print(f"# {sep}")

    print(f'\n{"Dataset":<20}', end='')
    for m in ['SimpleAvg', 'WeightedR2CV', 'WeightedR2CV2', 'LRStacking', 'RidgeStacking']:
        print(f'{m:>12}', end='')
    print(f'{"BestSingle":>12}  {"BestEnsemble":>12}  {"Gain":>8}')
    print(f'{"-" * 100}')

    for key in all_results:
        print(f'{key:<20}', end='')
        best_r2cv = -1
        best_method = ''
        for m in ['SimpleAvg', 'WeightedR2CV', 'WeightedR2CV2', 'LRStacking', 'RidgeStacking']:
            v = all_results[key][m]['R2CV']
            print(f'{v:>12.4f}', end='')
            if v > best_r2cv:
                best_r2cv = v
                best_method = m

        best_single = max(all_single_algo[key].values())
        gain = best_r2cv - best_single
        print(f'{best_single:>12.4f}  {best_method:>12}  {gain:>+8.4f}')

    # 保存结果
    output_path = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'ensemble_comparison.json')
    save_data = {
        'ensemble_results': all_results,
        'single_algo_r2cv': all_single_algo,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
