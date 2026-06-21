"""
正确的Stacking评估：训练集/测试集分离
========================================
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
from sklearn.model_selection import KFold, train_test_split
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


def get_oof_preds(X, y, params):
    """为单个算法生成OOF预测"""
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(n)

    for train_idx, val_idx in kf.split(X):
        Mdl = build_ann(params)
        Mdl.fit(X[train_idx], y[train_idx])
        oof[val_idx] = Mdl.predict(X[val_idx])

    return oof


def evaluate(preds, y):
    """评估预测结果"""
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, rmse, mae


def run_experiment():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    results_all = {}

    for dataset_key in DATASET_KEYS:
        print(f'\n{"=" * 60}')
        print(f'Dataset: {dataset_key}')
        print(f'{"=" * 60}')

        filename = DATASET_KEYS[dataset_key]
        X, y, _, _ = load_dataset(filename)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        # 划分训练集和测试集 (80/20)
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f'n_train={len(train_idx)}, n_test={len(test_idx)}')

        # 获取算法列表
        json_dataset = json_data.get(dataset_key, {})
        algo_list = []
        for json_key, algo_name in ALGO_MAP.items():
            if json_key in json_dataset and 'R2CV' in json_dataset[json_key]:
                algo_list.append((algo_name, json_dataset[json_key]))

        print(f'Algorithms: {[a[0] for a in algo_list]}')

        # ===== 方法1: 在训练集上做OOF，然后训练元学习器，在测试集上评估 =====
        print(f'\n--- Stacking with Train/Test Split ---')

        # 1. 为每个算法生成训练集的OOF预测
        oof_train = {}
        for algo_name, params in algo_list:
            oof_train[algo_name] = get_oof_preds(X_train, y_train, params)

        # 2. 为每个算法生成测试集预测（用全部训练数据训练的模型）
        test_preds = {}
        for algo_name, params in algo_list:
            Mdl = build_ann(params)
            Mdl.fit(X_train, y_train)
            test_preds[algo_name] = Mdl.predict(X_test)

        algo_names = [a[0] for a in algo_list]

        # 3. 各算法的单独表现
        print(f'\n  Single Algorithm Performance (on test set):')
        algo_test_r2 = {}
        for name in algo_names:
            r2, rmse, mae = evaluate(test_preds[name], y_test)
            algo_test_r2[name] = r2
            print(f'    {name:<12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

        best_single_name = max(algo_test_r2, key=algo_test_r2.get)
        best_single_r2 = algo_test_r2[best_single_name]
        print(f'  Best Single: {best_single_name} R2={best_single_r2:.4f}')

        # 4. LR Stacking
        # 训练集OOF堆叠
        X_train_stack = np.column_stack([oof_train[a] for a in algo_names])
        # 测试集预测堆叠
        X_test_stack = np.column_stack([test_preds[a] for a in algo_names])

        lr_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr_meta.fit(X_train_stack, y_train)
        lr_test_pred = lr_meta.predict(X_test_stack)
        lr_r2, lr_rmse, lr_mae = evaluate(lr_test_pred, y_test)
        lr_gain = lr_r2 - best_single_r2
        print(f'\n  LR Stacking:     R2={lr_r2:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  Gain={lr_gain:+.4f}')

        # 5. GB Stacking
        gb_meta = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=RANDOM_STATE))
        ])
        gb_meta.fit(X_train_stack, y_train)
        gb_test_pred = gb_meta.predict(X_test_stack)
        gb_r2, gb_rmse, gb_mae = evaluate(gb_test_pred, y_test)
        gb_gain = gb_r2 - best_single_r2
        print(f'  GB Stacking:    R2={gb_r2:.4f}  RMSE={gb_rmse:.4f}  MAE={gb_mae:.4f}  Gain={gb_gain:+.4f}')

        # 6. 简单平均
        simple_avg_pred = np.mean(X_test_stack, axis=1)
        sa_r2, sa_rmse, sa_mae = evaluate(simple_avg_pred, y_test)
        print(f'  SimpleAvg:      R2={sa_r2:.4f}  RMSE={sa_rmse:.4f}  MAE={sa_mae:.4f}')

        winner = 'GB' if gb_r2 > lr_r2 else 'LR'
        print(f'\n  Winner: {winner} Stacking')

        results_all[dataset_key] = {
            'best_single': best_single_r2,
            'lr_stacking': lr_r2,
            'gb_stacking': gb_r2,
            'simple_avg': sa_r2,
            'lr_gain': lr_gain,
            'gb_gain': gb_gain,
        }

    # ===== 汇总 =====
    print(f'\n\n{"=" * 80}')
    print(f'FINAL SUMMARY: Ensemble Methods (Test Set Evaluation)')
    print(f'{"=" * 80}')
    print(f'{"Dataset":<20} {"BestSingle":>10} {"SimpleAvg":>10} {"LR_Stack":>10} {"GB_Stack":>10} {"LR_Gain":>10} {"GB_Gain":>10}')
    print(f'{"-" * 80}')

    for key, r in results_all.items():
        print(f'{key:<20} {r["best_single"]:>10.4f} {r["simple_avg"]:>10.4f} {r["lr_stacking"]:>10.4f} {r["gb_stacking"]:>10.4f} {r["lr_gain"]:>+10.4f} {r["gb_gain"]:>+10.4f}')


if __name__ == '__main__':
    run_experiment()
