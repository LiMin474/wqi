"""
5种集成方法系统对比实验（快速版）
===================================
用保存的超参数重建ANN，跳过优化过程，快速生成结果。
基学习器：DE, SHADE, APSM-jSO, CMA-ES（最多4个，看JSON中哪些有数据）
"""

import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_results_v2.json')
N_FOLDS = 5
RANDOM_STATE = 1

# 数据集key映射
DATASET_KEYS = {
    '1_jajpur': '1_jajpur',
    '2_wqi_dataset': '2_wqi_dataset',
    '3_sample_dataset': '3_sample_dataset',
    '4_akh_wqi': '4_akh_wqi',
}

# JSON中的算法名映射到我们要用的名字
ALGO_MAP = {
    'DE': 'DE',
    'SHADE': 'SHADE',
    'APSM-jSO (2023)': 'APSM-jSO',
    'CMA-ES': 'CMA-ES',
    'Bayesian': 'Bayesian',
}


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    return data['X'], data['y'], str(data['name']), str(data['target_name'])


def build_ann(params):
    n_layers = params['NumLayers']
    layer1 = params['Layer_1']
    layer2 = params['Layer_2']
    activation = params['Activation']
    alpha = params.get('Alpha', 1e-6)
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
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


def get_oof_preds_from_json(X, y, json_data, algo_keys):
    """
    用JSON中的超参数，对每个算法生成5折out-of-fold预测。
    使用JSON中记录的随机种子来重建模型。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = {}
    algo_r2cv = {}

    for algo_name, json_key in algo_keys.items():
        if json_key not in json_data:
            print(f'    [WARN] {json_key} not found in JSON, skipping')
            continue

        params = json_data[json_key]
        if 'R2CV' not in params:
            print(f'    [WARN] {json_key} has no R2CV, skipping')
            continue

        oof = np.zeros(n)
        fold_r2_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 用JSON中的超参数构建ANN
            Mdl = build_ann(params)
            Mdl.fit(X_tr, y_tr)
            oof[val_idx] = Mdl.predict(X_va)

            sst = np.sum((y_va - np.mean(y_va)) ** 2)
            ssr = np.sum((y_va - oof[val_idx]) ** 2)
            r2 = 1 - ssr / sst if sst > 0 else 0
            fold_r2_list.append(r2)

        r2cv = float(np.mean(fold_r2_list))
        oof_preds[algo_name] = oof
        algo_r2cv[algo_name] = r2cv
        print(f'    {algo_name:<12s}: R2CV={r2cv:.4f}  (folds: {[f"{r:.4f}" for r in fold_r2_list]})')

    return oof_preds, algo_r2cv


def ensemble_simple_avg(oof_preds, y, algo_names):
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    return np.mean(preds_mat, axis=1)


def ensemble_weighted_r2cv(oof_preds, y, algo_names, algo_r2cv):
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    weights = np.array([algo_r2cv[a] for a in algo_names])
    weights = weights / weights.sum()
    return np.average(preds_mat, axis=1, weights=weights)


def ensemble_weighted_r2cv2(oof_preds, y, algo_names, algo_r2cv):
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    weights = np.array([algo_r2cv[a] ** 2 for a in algo_names])
    weights = weights / weights.sum()
    return np.average(preds_mat, axis=1, weights=weights)


def ensemble_lr_stacking(oof_preds, y, algo_names):
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    meta.fit(preds_mat, y)
    return meta.predict(preds_mat)


def ensemble_ridge_stacking(oof_preds, y, algo_names):
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    meta = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    meta.fit(preds_mat, y)
    return meta.predict(preds_mat)


def evaluate(preds, y):
    r2 = r2_score(y, preds)
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


def run_for_dataset(dataset_key, json_data):
    """对单个数据集跑5种集成方法"""
    print(f'\n{"=" * 60}')
    print(f'Dataset: {dataset_key}')
    print(f'{"=" * 60}')

    filename = DATASET_KEYS[dataset_key]
    X, y, dataset_name, target_name = load_dataset(filename)
    print(f'n={len(y)}, features={X.shape[1]}, target={target_name}')
    print(f'target: mean={y.mean():.3f}, std={y.std():.3f}')

    # 找出JSON中可用的算法
    algo_keys = {}
    available_algos = list(ALGO_MAP.keys())
    json_dataset = json_data.get(dataset_key, {})
    for json_key in available_algos:
        if json_key in json_dataset:
            algo_keys[ALGO_MAP[json_key]] = json_key

    print(f'\n--- Available algorithms: {list(algo_keys.keys())} ---')
    oof_preds, algo_r2cv = get_oof_preds_from_json(X, y, json_dataset, algo_keys)
    algo_names = list(oof_preds.keys())

    if len(algo_names) < 2:
        print('[ERROR] Less than 2 algorithms available, cannot ensemble')
        return None, None, None

    print(f'\n--- Ensemble Methods Comparison ---')
    results = {}

    # 方法1: 简单平均
    preds1 = ensemble_simple_avg(oof_preds, y, algo_names)
    r2, r2cv, rmse, mae = evaluate(preds1, y)
    results['SimpleAvg'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [1] SimpleAvg     R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法2: 加权R2CV
    preds2 = ensemble_weighted_r2cv(oof_preds, y, algo_names, algo_r2cv)
    r2, r2cv, rmse, mae = evaluate(preds2, y)
    results['WeightedR2CV'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [2] WeightedR2CV  R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法3: 加权R2CV^2
    preds3 = ensemble_weighted_r2cv2(oof_preds, y, algo_names, algo_r2cv)
    r2, r2cv, rmse, mae = evaluate(preds3, y)
    results['WeightedR2CV2'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [3] WeightedR2CV2 R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法4: LR Stacking
    preds4 = ensemble_lr_stacking(oof_preds, y, algo_names)
    r2, r2cv, rmse, mae = evaluate(preds4, y)
    results['LRStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [4] LRStacking   R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 方法5: Ridge Stacking
    preds5 = ensemble_ridge_stacking(oof_preds, y, algo_names)
    r2, r2cv, rmse, mae = evaluate(preds5, y)
    results['RidgeStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    print(f'  [5] RidgeStacking R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 找最优
    best_method = max(results, key=lambda k: results[k]['R2CV'])
    best_r2cv = results[best_method]['R2CV']
    best_single = max(algo_r2cv, key=lambda k: algo_r2cv[k])
    best_single_r2cv = algo_r2cv[best_single]
    improvement = best_r2cv - best_single_r2cv
    print(f'\n  >>> Best ensemble: {best_method} (R2CV={best_r2cv:.4f})')
    print(f'  >>> Best single:  {best_single} (R2CV={best_single_r2cv:.4f})')
    print(f'  >>> Gain: +{improvement:.4f}')

    return results, algo_r2cv, best_method


def main():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f'Loaded {len(json_data)} datasets from JSON')

    all_results = {}
    all_single = {}

    for dataset_key in DATASET_KEYS.keys():
        results, algo_r2cv, best_method = run_for_dataset(dataset_key, json_data)
        if results:
            all_results[dataset_key] = results
            all_single[dataset_key] = algo_r2cv

    # ===== 汇总表 =====
    print(f'\n\n{"=" * 80}')
    print(f'FINAL SUMMARY: 5 Ensemble Methods x {len(all_results)} Datasets')
    print(f'{"=" * 80}')
    methods = ['SimpleAvg', 'WeightedR2CV', 'WeightedR2CV2', 'LRStacking', 'RidgeStacking']
    header = f'{"Dataset":<20}'
    for m in methods:
        header += f'{m:>12}'
    header += f'{"BestSingle":>12}  {"BestEns":>12}  {"Gain":>8}'
    print(header)
    print(f'{"-" * 100}')

    for key in all_results:
        row = f'{key:<20}'
        best_r2cv = -1
        best_m = ''
        for m in methods:
            v = all_results[key][m]['R2CV']
            row += f'{v:>12.4f}'
            if v > best_r2cv:
                best_r2cv = v
                best_m = m
        bs = max(all_single[key].values())
        gain = best_r2cv - bs
        row += f'{bs:>12.4f}  {best_m:>12}  {gain:>+8.4f}'
        print(row)

    # 保存结果
    out_path = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'ensemble_comparison.json')
    save_data = {'ensemble_results': all_results, 'single_algo_r2cv': all_single}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
