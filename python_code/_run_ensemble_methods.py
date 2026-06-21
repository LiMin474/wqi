"""
多种集成方法对比实验
====================
测试 Blending、多层 Stacking、RF 元学习器等方法
"""

import numpy as np
import os
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_results_v2.json')
N_FOLDS = 5
RANDOM_STATE = 1

DATASET_KEYS = {
    '4_akh_wqi': '4_akh_wqi',  # 重点测试 AKH（最难的数据集）
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

    for dataset_key in DATASET_KEYS:
        print(f'\n{"=" * 70}')
        print(f'Dataset: {dataset_key}')
        print(f'{"=" * 70}')

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

        algo_names = [a[0] for a in algo_list]
        print(f'Algorithms: {algo_names}')

        # 为每个算法生成训练集的OOF预测
        print(f'\nGenerating OOF predictions...')
        oof_train = {}
        for algo_name, params in algo_list:
            oof_train[algo_name] = get_oof_preds(X_train, y_train, params)
            print(f'  {algo_name} done')

        # 为每个算法生成测试集预测
        test_preds = {}
        for algo_name, params in algo_list:
            Mdl = build_ann(params)
            Mdl.fit(X_train, y_train)
            test_preds[algo_name] = Mdl.predict(X_test)

        # 各算法的单独表现
        print(f'\n--- Single Algorithm Performance ---')
        algo_test_r2 = {}
        for name in algo_names:
            r2, rmse, mae = evaluate(test_preds[name], y_test)
            algo_test_r2[name] = r2
            print(f'  {name:<12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

        best_single_name = max(algo_test_r2, key=algo_test_r2.get)
        best_single_r2 = algo_test_r2[best_single_name]
        print(f'\nBest Single: {best_single_name} R2={best_single_r2:.4f}')

        # 特征矩阵
        X_train_stack = np.column_stack([oof_train[a] for a in algo_names])
        X_test_stack = np.column_stack([test_preds[a] for a in algo_names])

        # ===== 集成方法对比 =====
        print(f'\n--- Ensemble Methods Comparison ---')
        results = []

        # 1. 简单平均
        sa_pred = np.mean(X_test_stack, axis=1)
        sa_r2, sa_rmse, sa_mae = evaluate(sa_pred, y_test)
        results.append(('SimpleAvg', sa_r2, sa_rmse, sa_mae, sa_r2 - best_single_r2))
        print(f'  [1] SimpleAvg       R2={sa_r2:.4f}  RMSE={sa_rmse:.4f}  MAE={sa_mae:.4f}  Gain={sa_r2 - best_single_r2:+.4f}')

        # 2. 加权平均（按单算法R2）
        weights = np.array([algo_test_r2[a] for a in algo_names])
        weights = weights / weights.sum()
        wa_pred = np.average(X_test_stack, axis=1, weights=weights)
        wa_r2, wa_rmse, wa_mae = evaluate(wa_pred, y_test)
        results.append(('WeightedAvg', wa_r2, wa_rmse, wa_mae, wa_r2 - best_single_r2))
        print(f'  [2] WeightedAvg    R2={wa_r2:.4f}  RMSE={wa_rmse:.4f}  MAE={wa_mae:.4f}  Gain={wa_r2 - best_single_r2:+.4f}')

        # 3. LR Stacking
        lr_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr_meta.fit(X_train_stack, y_train)
        lr_pred = lr_meta.predict(X_test_stack)
        lr_r2, lr_rmse, lr_mae = evaluate(lr_pred, y_test)
        results.append(('LRStacking', lr_r2, lr_rmse, lr_mae, lr_r2 - best_single_r2))
        print(f'  [3] LRStacking     R2={lr_r2:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  Gain={lr_r2 - best_single_r2:+.4f}')

        # 4. Ridge Stacking
        ridge_meta = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        ridge_meta.fit(X_train_stack, y_train)
        ridge_pred = ridge_meta.predict(X_test_stack)
        ridge_r2, ridge_rmse, ridge_mae = evaluate(ridge_pred, y_test)
        results.append(('RidgeStacking', ridge_r2, ridge_rmse, ridge_mae, ridge_r2 - best_single_r2))
        print(f'  [4] RidgeStacking  R2={ridge_r2:.4f}  RMSE={ridge_rmse:.4f}  MAE={ridge_mae:.4f}  Gain={ridge_r2 - best_single_r2:+.4f}')

        # 5. RF Stacking
        rf_meta = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))])
        rf_meta.fit(X_train_stack, y_train)
        rf_pred = rf_meta.predict(X_test_stack)
        rf_r2, rf_rmse, rf_mae = evaluate(rf_pred, y_test)
        results.append(('RFStacking', rf_r2, rf_rmse, rf_mae, rf_r2 - best_single_r2))
        print(f'  [5] RFStacking     R2={rf_r2:.4f}  RMSE={rf_rmse:.4f}  MAE={rf_mae:.4f}  Gain={rf_r2 - best_single_r2:+.4f}')

        # 6. GB Stacking
        gb_meta = Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE))])
        gb_meta.fit(X_train_stack, y_train)
        gb_pred = gb_meta.predict(X_test_stack)
        gb_r2, gb_rmse, gb_mae = evaluate(gb_pred, y_test)
        results.append(('GBStacking', gb_r2, gb_rmse, gb_mae, gb_r2 - best_single_r2))
        print(f'  [6] GBStacking     R2={gb_r2:.4f}  RMSE={gb_rmse:.4f}  MAE={gb_mae:.4f}  Gain={gb_r2 - best_single_r2:+.4f}')

        # 7. 多层 Stacking（LR -> GB -> LR）
        # 第一层：LR
        lr1_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr1_meta.fit(X_train_stack, y_train)
        train_lr1_pred = lr1_meta.predict(X_train_stack)
        test_lr1_pred = lr1_meta.predict(X_test_stack)

        # 第二层：把LR输出和原始特征拼接
        X_train_stack2 = np.column_stack([X_train_stack, train_lr1_pred])
        X_test_stack2 = np.column_stack([X_test_stack, test_lr1_pred])

        # 第二层：GB
        gb2_meta = Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE))])
        gb2_meta.fit(X_train_stack2, y_train)
        train_gb2_pred = gb2_meta.predict(X_train_stack2)
        test_gb2_pred = gb2_meta.predict(X_test_stack2)

        # 第三层：LR
        X_train_stack3 = np.column_stack([X_train_stack2, train_gb2_pred])
        X_test_stack3 = np.column_stack([X_test_stack2, test_gb2_pred])

        lr3_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr3_meta.fit(X_train_stack3, y_train)
        ml_pred = lr3_meta.predict(X_test_stack3)
        ml_r2, ml_rmse, ml_mae = evaluate(ml_pred, y_test)
        results.append(('MultiLayerStack', ml_r2, ml_rmse, ml_mae, ml_r2 - best_single_r2))
        print(f'  [7] MultiLayerStack R2={ml_r2:.4f}  RMSE={ml_rmse:.4f}  MAE={ml_mae:.4f}  Gain={ml_r2 - best_single_r2:+.4f}')

        # 8. Blending（不用OOF，直接用验证集）
        # 把训练集分成两部分
        train_idx2, val_idx2 = train_test_split(train_idx, test_size=0.2, random_state=RANDOM_STATE)
        X_tr, X_val = X[train_idx2], X[val_idx2]
        y_tr, y_val = y[train_idx2], y[val_idx2]

        # 在训练集上训练模型，在验证集和测试集上预测
        val_preds = {}
        test_preds2 = {}
        for algo_name, params in algo_list:
            Mdl = build_ann(params)
            Mdl.fit(X_tr, y_tr)
            val_preds[algo_name] = Mdl.predict(X_val)
            test_preds2[algo_name] = Mdl.predict(X_test)

        X_val_stack = np.column_stack([val_preds[a] for a in algo_names])
        X_test_stack2 = np.column_stack([test_preds2[a] for a in algo_names])

        # 用验证集训练元学习器
        blend_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        blend_meta.fit(X_val_stack, y_val)
        blend_pred = blend_meta.predict(X_test_stack2)
        blend_r2, blend_rmse, blend_mae = evaluate(blend_pred, y_test)
        results.append(('Blending', blend_r2, blend_rmse, blend_mae, blend_r2 - best_single_r2))
        print(f'  [8] Blending       R2={blend_r2:.4f}  RMSE={blend_rmse:.4f}  MAE={blend_mae:.4f}  Gain={blend_r2 - best_single_r2:+.4f}')

        # ===== 汇总 =====
        print(f'\n{"=" * 70}')
        print(f'SUMMARY for {dataset_key}')
        print(f'{"=" * 70}')
        print(f'Best Single: {best_single_name} R2={best_single_r2:.4f}')
        print(f'{"-" * 70}')
        print(f'{"Method":<15} {"R2":>10} {"RMSE":>10} {"MAE":>10} {"Gain":>10}')
        print(f'{"-" * 70}')

        # 按 Gain 排序
        results.sort(key=lambda x: x[4], reverse=True)
        for method, r2, rmse, mae, gain in results:
            print(f'{method:<15} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {gain:>+10.4f}')

        best_method = results[0][0]
        best_gain = results[0][4]
        print(f'\nBest Ensemble Method: {best_method} (Gain: {best_gain:+.4f})')


if __name__ == '__main__':
    run_experiment()
