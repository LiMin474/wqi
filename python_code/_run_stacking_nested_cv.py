"""
正确的 Stacking 评估：嵌套CV
===============================
Level 2 元学习器也必须用CV评估，避免过拟合
"""

import numpy as np
import os
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_results_v2.json')
N_FOLDS = 5
RANDOM_STATE = 1

DATASET_KEYS = {
    '1_jajpur': '1_jajpur',
    '2_wqi_dataset': '2_wqi_dataset',
    '3_sample_dataset': '3_sample_dataset',
    '4_akh_wqi': '4_akh_wqi',
}

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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = {}
    algo_r2cv = {}

    for algo_name, json_key in algo_keys.items():
        if json_key not in json_data:
            continue

        params = json_data[json_key]
        if 'R2CV' not in params:
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

    return oof_preds, algo_r2cv


def evaluate_cv(preds, y):
    """正确的CV评估：对预测值做5折CV"""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_r2_list = []
    for train_idx, val_idx in kf.split(preds):
        sst = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
        ssr = np.sum((y[val_idx] - preds[val_idx]) ** 2)
        fold_r2_list.append(1 - ssr / sst if sst > 0 else 0)
    return float(np.mean(fold_r2_list))


def stacking_with_nested_cv(oof_preds, y, algo_names, meta_learner_type='lr'):
    """
    正确的Stacking：用嵌套CV评估元学习器
    ========================
    外层循环：把数据分成5折
    内层：对每一折，用其余4折训练元学习器，预测这一折

    这样可以避免元学习器在训练数据上过拟合
    """
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_meta = np.zeros(n)
    fold_r2_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(oof_preds)):
        # 训练数据的OOF预测
        X_train = np.column_stack([oof_preds[a][train_idx] for a in algo_names])
        y_train = y[train_idx]

        # 验证数据的OOF预测
        X_val = np.column_stack([oof_preds[a][val_idx] for a in algo_names])
        y_val = y[val_idx]

        # 训练元学习器
        if meta_learner_type == 'lr':
            meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        else:  # gb
            meta = Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
            ))])

        meta.fit(X_train, y_train)
        oof_meta[val_idx] = meta.predict(X_val)

        # 计算这一折的R²
        sst = np.sum((y_val - np.mean(y_val)) ** 2)
        ssr = np.sum((y_val - oof_meta[val_idx]) ** 2)
        fold_r2 = 1 - ssr / sst if sst > 0 else 0
        fold_r2_list.append(fold_r2)

    r2cv = float(np.mean(fold_r2_list))
    rmse = np.sqrt(mean_squared_error(y, oof_meta))
    mae = mean_absolute_error(y, oof_meta)
    return r2cv, rmse, mae, fold_r2_list


def run_for_dataset(dataset_key, json_data):
    print(f'\n{"=" * 60}')
    print(f'Dataset: {dataset_key}')
    print(f'{"=" * 60}')

    filename = DATASET_KEYS[dataset_key]
    X, y, dataset_name, target_name = load_dataset(filename)
    print(f'n={len(y)}, features={X.shape[1]}, target={target_name}')

    algo_keys = {}
    available_algos = list(ALGO_MAP.keys())
    json_dataset = json_data.get(dataset_key, {})
    for json_key in available_algos:
        if json_key in json_dataset:
            algo_keys[ALGO_MAP[json_key]] = json_key

    print(f'\n--- Available algorithms: {list(algo_keys.keys())} ---')
    oof_preds, algo_r2cv = get_oof_preds_from_json(X, y, json_dataset, algo_keys)
    algo_names = list(oof_preds.keys())

    print(f'\n--- Base model R2CV ---')
    for name in algo_names:
        print(f'    {name:<12s}: {algo_r2cv[name]:.4f}')

    best_single = max(algo_r2cv.values())
    print(f'\nBest Single R2CV: {best_single:.4f}')

    # 正确的嵌套CV评估
    print(f'\n--- Stacking with Nested CV (Correct Method) ---')

    # LR Stacking
    lr_r2cv, lr_rmse, lr_mae, lr_folds = stacking_with_nested_cv(oof_preds, y, algo_names, 'lr')
    lr_gain = lr_r2cv - best_single
    print(f'  LR Stacking:  R2CV={lr_r2cv:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  Gain={lr_gain:+.4f}')
    print(f'    Fold R2: {[f"{r:.4f}" for r in lr_folds]}')

    # GB Stacking
    gb_r2cv, gb_rmse, gb_mae, gb_folds = stacking_with_nested_cv(oof_preds, y, algo_names, 'gb')
    gb_gain = gb_r2cv - best_single
    print(f'  GB Stacking:  R2CV={gb_r2cv:.4f}  RMSE={gb_rmse:.4f}  MAE={gb_mae:.4f}  Gain={gb_gain:+.4f}')
    print(f'    Fold R2: {[f"{r:.4f}" for r in gb_folds]}')

    winner = 'GB' if gb_r2cv > lr_r2cv else 'LR'
    print(f'\n  Winner: {winner} Stacking')

    return {
        'best_single': best_single,
        'lr_r2cv': lr_r2cv, 'lr_rmse': lr_rmse, 'lr_mae': lr_mae, 'lr_gain': lr_gain,
        'gb_r2cv': gb_r2cv, 'gb_rmse': gb_rmse, 'gb_mae': gb_mae, 'gb_gain': gb_gain,
    }


def main():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    all_results = {}

    for dataset_key in DATASET_KEYS.keys():
        results = run_for_dataset(dataset_key, json_data)
        all_results[dataset_key] = results

    # 汇总
    print(f'\n\n{"=" * 80}')
    print(f'FINAL SUMMARY: LR vs GB Stacking (Nested CV - Correct Evaluation)')
    print(f'{"=" * 80}')
    print(f'{"Dataset":<20} {"BestSingle":>10} {"LR_Stacking":>10} {"GB_Stacking":>10} {"LR_Gain":>10} {"GB_Gain":>10} {"Winner":>8}')
    print(f'{"-" * 80}')

    for key, r in all_results.items():
        winner = 'GB' if r['gb_r2cv'] > r['lr_r2cv'] else 'LR'
        print(f'{key:<20} {r["best_single"]:>10.4f} {r["lr_r2cv"]:>10.4f} {r["gb_r2cv"]:>10.4f} {r["lr_gain"]:>+10.4f} {r["gb_gain"]:>+10.4f} {winner:>8}')


if __name__ == '__main__':
    main()
