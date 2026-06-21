"""
XGBoost Stacking vs LR Stacking 对比实验
=========================================
用已保存的超参数重建ANN，对比两种元学习器的效果。
"""

import numpy as np
import os
import sys
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBRegressor = GradientBoostingRegressor  # 用 GradientBoosting 替代
    HAS_XGB = True
    print('[INFO] XGBoost not installed, using GradientBoostingRegressor as alternative')

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

# JSON中的算法名映射
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
    """用JSON中的超参数，对每个算法生成5折out-of-fold预测"""
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
        print(f'    {algo_name:<12s}: R2CV={r2cv:.4f}')

    return oof_preds, algo_r2cv


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


def ensemble_lr_stacking(oof_preds, y, algo_names):
    """LR Stacking"""
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    meta.fit(preds_mat, y)
    return meta.predict(preds_mat)


def ensemble_xgboost_stacking(oof_preds, y, algo_names):
    """XGBoost Stacking (or GradientBoosting as fallback)"""
    if not HAS_XGB:
        return None, None
    
    preds_mat = np.column_stack([oof_preds[a] for a in algo_names])
    
    # 使用兼容参数（GradientBoosting 不支持 verbosity）
    meta = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    
    meta.fit(preds_mat, y)
    preds = meta.predict(preds_mat)
    
    # 获取特征重要性
    importance = meta.feature_importances_
    return preds, importance


def run_for_dataset(dataset_key, json_data):
    """对单个数据集对比 LR vs XGBoost Stacking"""
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
        print('[ERROR] Less than 2 algorithms available')
        return None

    print(f'\n--- Ensemble Methods Comparison ---')
    results = {}

    # 最优单算法
    best_single = max(algo_r2cv, key=lambda k: algo_r2cv[k])
    best_single_r2cv = algo_r2cv[best_single]
    print(f'  [Base] Best Single: {best_single} R2CV={best_single_r2cv:.4f}')

    # LR Stacking
    preds_lr = ensemble_lr_stacking(oof_preds, y, algo_names)
    r2, r2cv, rmse, mae = evaluate(preds_lr, y)
    results['LRStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
    gain_lr = r2cv - best_single_r2cv
    print(f'  [LR]   LRStacking   R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  Gain={gain_lr:+.4f}')

    # XGBoost Stacking
    if HAS_XGB:
        preds_xgb, importance = ensemble_xgboost_stacking(oof_preds, y, algo_names)
        if preds_xgb is not None:
            r2, r2cv, rmse, mae = evaluate(preds_xgb, y)
            results['XGBoostStacking'] = {'R2': r2, 'R2CV': r2cv, 'RMSE': rmse, 'MAE': mae}
            gain_xgb = r2cv - best_single_r2cv
            print(f'  [XGB]  XGBoostStack R2={r2:.4f}  R2CV={r2cv:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  Gain={gain_xgb:+.4f}')
            
            # 打印特征重要性
            print(f'\n  XGBoost Feature Importance (algorithm weights):')
            for i, name in enumerate(algo_names):
                print(f'    {name:<12s}: {importance[i]:.4f}')

            # 对比
            if r2cv > results['LRStacking']['R2CV']:
                print(f'\n  >>> XGBoost Stacking BETTER than LR Stacking!')
            else:
                print(f'\n  >>> LR Stacking still better')

    return results, algo_r2cv


def main():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f'Loaded {len(json_data)} datasets from JSON')

    all_results = {}
    all_single = {}

    for dataset_key in DATASET_KEYS.keys():
        results, algo_r2cv = run_for_dataset(dataset_key, json_data)
        if results:
            all_results[dataset_key] = results
            all_single[dataset_key] = algo_r2cv

    # ===== 汇总表 =====
    print(f'\n\n{"=" * 80}')
    print(f'FINAL SUMMARY: LR Stacking vs XGBoost Stacking')
    print(f'{"=" * 80}')
    
    header = f'{"Dataset":<20} {"BestSingle":>12} {"LRStacking":>12} {"XGBoost":>12} {"LR_Gain":>10} {"XGB_Gain":>10} {"Winner":>10}'
    print(header)
    print(f'{"-" * 90}')

    for key in all_results:
        bs = max(all_single[key].values())
        lr_r2cv = all_results[key]['LRStacking']['R2CV']
        lr_gain = lr_r2cv - bs
        
        if 'XGBoostStacking' in all_results[key]:
            xgb_r2cv = all_results[key]['XGBoostStacking']['R2CV']
            xgb_gain = xgb_r2cv - bs
            winner = 'XGBoost' if xgb_r2cv > lr_r2cv else 'LR'
            print(f'{key:<20} {bs:>12.4f} {lr_r2cv:>12.4f} {xgb_r2cv:>12.4f} {lr_gain:>+10.4f} {xgb_gain:>+10.4f} {winner:>10}')
        else:
            print(f'{key:<20} {bs:>12.4f} {lr_r2cv:>12.4f} {"N/A":>12} {lr_gain:>+10.4f} {"N/A":>10} {"LR":>10}')

    # 保存结果
    out_path = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'xgboost_stacking_comparison.json')
    save_data = {'ensemble_results': all_results, 'single_algo_r2cv': all_single}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()