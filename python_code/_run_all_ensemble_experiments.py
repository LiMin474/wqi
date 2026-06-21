"""
完整集成实验：全部 4 数据集 × 多种集成方法
=========================================
包括：SimpleAvg, WeightedAvg, LR Stacking, Ridge Stacking, 
      RF Stacking, GB Stacking, FuzzyEnsemble, BMA, AdaBoost
"""

import numpy as np
import os
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
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
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(n)
    for train_idx, val_idx in kf.split(X):
        Mdl = build_ann(params)
        Mdl.fit(X[train_idx], y[train_idx])
        oof[val_idx] = Mdl.predict(X[val_idx])
    return oof


def evaluate(preds, y):
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, rmse, mae


class FuzzyEnsemble:
    def __init__(self, base_models, eta=1.0):
        self.base_models = base_models
        self.eta = eta
        self.weights_ = None
    
    def _fuzzy_membership(self, x, a, b, c):
        if x <= a: return 0.0
        elif a < x <= b: return (x - a) / (b - a)
        elif b < x <= c: return (c - x) / (c - b)
        else: return 0.0
    
    def _fuzzy_inference(self, error_rate):
        mu = []
        mu.append(self._fuzzy_membership(error_rate, 0, 0, 0.1))
        mu.append(self._fuzzy_membership(error_rate, 0, 0.1, 0.2))
        mu.append(self._fuzzy_membership(error_rate, 0.1, 0.2, 0.3))
        mu.append(self._fuzzy_membership(error_rate, 0.2, 0.3, 0.4))
        mu.append(self._fuzzy_membership(error_rate, 0.3, 0.4, 0.5))
        mu.append(self._fuzzy_membership(error_rate, 0.4, 0.5, 0.7))
        mu.append(self._fuzzy_membership(error_rate, 0.5, 0.7, 1.0))
        output_centers = [0.95, 0.85, 0.75, 0.60, 0.45, 0.30, 0.15]
        numerator = sum(mu[i] * output_centers[i] for i in range(7))
        denominator = sum(mu[i] for i in range(7))
        return numerator / denominator if denominator > 0 else 0.5
    
    def fit(self, X, y):
        n = len(y)
        n_models = len(self.base_models)
        predictions = np.zeros((n, n_models))
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        tau_err = np.zeros(n_models)
        delta_err = np.zeros(n_models)
        for i in range(n_models):
            errors = np.abs(predictions[:, i] - y)
            tau_err[i] = np.mean(errors)
            delta_err[i] = np.std(errors)
        hard_samples = np.zeros((n, n_models), dtype=bool)
        for i in range(n_models):
            threshold = tau_err[i] + self.eta * delta_err[i]
            hard_samples[:, i] = np.abs(predictions[:, i] - y) > threshold
        error_rates = np.zeros(n_models)
        for i in range(n_models):
            if np.any(hard_samples[:, i]):
                hard_errors = np.abs(predictions[hard_samples[:, i], i] - y[hard_samples[:, i]])
                error_rates[i] = np.mean(hard_errors) / (np.max(y) - np.min(y))
            else:
                error_rates[i] = 0.1
        self.weights_ = np.array([self._fuzzy_inference(rate) for rate in error_rates])
        self.weights_ = self.weights_ / np.sum(self.weights_)
        return self
    
    def predict(self, X):
        n = len(X)
        n_models = len(self.base_models)
        predictions = np.zeros((n, n_models))
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        return np.dot(predictions, self.weights_)


def run_experiment():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    all_results = {}

    for dataset_key in DATASET_KEYS:
        print(f'\n{"=" * 70}')
        print(f'Dataset: {dataset_key}')
        print(f'{"=" * 70}')

        filename = DATASET_KEYS[dataset_key]
        X, y, _, _ = load_dataset(filename)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f'n_train={len(train_idx)}, n_test={len(test_idx)}')

        json_dataset = json_data.get(dataset_key, {})
        algo_list = []
        for json_key, algo_name in ALGO_MAP.items():
            if json_key in json_dataset and 'R2CV' in json_dataset[json_key]:
                algo_list.append((algo_name, json_dataset[json_key]))

        algo_names = [a[0] for a in algo_list]
        print(f'Algorithms: {algo_names}')

        # 训练基模型
        print(f'\nTraining base models...')
        base_models = []
        for algo_name, params in algo_list:
            Mdl = build_ann(params)
            Mdl.fit(X_train, y_train)
            base_models.append((algo_name, Mdl))

        # 测试集预测
        test_preds_dict = {}
        for name, model in base_models:
            test_preds_dict[name] = model.predict(X_test)

        # 单算法性能
        print(f'\n  Single Algorithm Performance:')
        algo_r2 = {}
        for name in algo_names:
            r2, rmse, mae = evaluate(test_preds_dict[name], y_test)
            algo_r2[name] = r2
            print(f'    {name:<12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

        best_name = max(algo_r2, key=algo_r2.get)
        best_r2 = algo_r2[best_name]
        print(f'\n  Best Single: {best_name} R2={best_r2:.4f}')

        # 生成 OOF 预测（用于 Stacking）
        oof_train = {}
        for algo_name, params in algo_list:
            oof_train[algo_name] = get_oof_preds(X_train, y_train, params)

        X_train_stack = np.column_stack([oof_train[a] for a in algo_names])
        X_test_stack = np.column_stack([test_preds_dict[a] for a in algo_names])

        # ===== 集成方法 =====
        print(f'\n  Ensemble Methods:')
        results = {}

        # 1. SimpleAvg
        sa_pred = np.mean(X_test_stack, axis=1)
        results['SimpleAvg'] = evaluate(sa_pred, y_test)

        # 2. WeightedAvg
        weights = np.array([algo_r2[name] for name in algo_names])
        weights = np.maximum(weights, 0) / np.sum(np.maximum(weights, 0))
        wa_pred = np.average(X_test_stack, axis=1, weights=weights)
        results['WeightedAvg'] = evaluate(wa_pred, y_test)

        # 3. LR Stacking
        lr_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr_meta.fit(X_train_stack, y_train)
        lr_pred = lr_meta.predict(X_test_stack)
        results['LRStacking'] = evaluate(lr_pred, y_test)

        # 4. Ridge Stacking
        ridge_meta = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        ridge_meta.fit(X_train_stack, y_train)
        ridge_pred = ridge_meta.predict(X_test_stack)
        results['RidgeStacking'] = evaluate(ridge_pred, y_test)

        # 5. RF Stacking
        rf_meta = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))])
        rf_meta.fit(X_train_stack, y_train)
        rf_pred = rf_meta.predict(X_test_stack)
        results['RFStacking'] = evaluate(rf_pred, y_test)

        # 6. GB Stacking
        gb_meta = Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE))])
        gb_meta.fit(X_train_stack, y_train)
        gb_pred = gb_meta.predict(X_test_stack)
        results['GBStacking'] = evaluate(gb_pred, y_test)

        # 7. Fuzzy Ensemble
        fuzzy = FuzzyEnsemble(base_models, eta=1.0)
        fuzzy.fit(X_train, y_train)
        fuzzy_pred = fuzzy.predict(X_test)
        results['FuzzyEnsemble'] = evaluate(fuzzy_pred, y_test)

        # 8. BMA (Bayesian Model Averaging)
        bma_weights = np.array([max(algo_r2[name], 0) for name in algo_names])
        bma_weights = bma_weights / np.sum(bma_weights)
        bma_pred = np.average(X_test_stack, axis=1, weights=bma_weights)
        results['BMA'] = evaluate(bma_pred, y_test)

        # 9. AdaBoost (直接用原始特征)
        ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=RANDOM_STATE)
        ada.fit(X_train_stack, y_train)
        ada_pred = ada.predict(X_test_stack)
        results['AdaBoost'] = evaluate(ada_pred, y_test)

        # 计算最佳增益
        best_gain = max(r2 - best_r2 for _, (r2, _, _) in results.items())
        
        # 打印结果
        print(f'\n  {"Method":<15} {"R2":>10} {"RMSE":>10} {"MAE":>10} {"Gain":>10}')
        print(f'  {"-" * 55}')
        
        sorted_results = sorted(results.items(), key=lambda x: x[1][0] - best_r2, reverse=True)
        for name, (r2, rmse, mae) in sorted_results:
            gain = r2 - best_r2
            marker = ' *' if abs(gain - best_gain) < 0.0001 else ''
            print(f'  {name:<15} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {gain:>+10.4f}{marker}')

        # 存储结果
        all_results[dataset_key] = {
            'best_single_name': best_name,
            'best_single_r2': best_r2,
            'single_results': {name: {'r2': algo_r2[name]} for name in algo_names},
            'ensemble_results': {name: {'r2': r2, 'rmse': rmse, 'mae': mae, 'gain': r2 - best_r2}
                                for name, (r2, rmse, mae) in results.items()}
        }

    # ===== 最终汇总表 =====
    print('\n\n')
    print('=' * 80)
    print('FINAL SUMMARY: All Datasets × All Ensemble Methods')
    print('=' * 80)
    
    print('\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'Dataset', 'BestSingle', 'SimpleAvg', 'WeightedAvg', 'LRStack', 'RidgeStack', 'Fuzzy', 'AdaBoost'))
    print('-' * 90)
    
    for dk in DATASET_KEYS:
        res = all_results[dk]
        best = res['best_single_r2']
        sa = res['ensemble_results']['SimpleAvg']['r2']
        wa = res['ensemble_results']['WeightedAvg']['r2']
        lr = res['ensemble_results']['LRStacking']['r2']
        ri = res['ensemble_results']['RidgeStacking']['r2']
        fu = res['ensemble_results']['FuzzyEnsemble']['r2']
        ad = res['ensemble_results']['AdaBoost']['r2']
        print(f'{dk:<20} {best:>10.4f} {sa:>10.4f} {wa:>10.4f} {lr:>10.4f} {ri:>10.4f} {fu:>10.4f} {ad:>10.4f}')

    # Gain 汇总
    print('\n\n')
    print('-' * 40, 'Gain Summary', '-' * 40)
    print('{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'Dataset', 'SA_Gain', 'WA_Gain', 'LR_Gain', 'Ridge_Gain', 'Fuzzy_Gain', 'Ada_Gain'))
    print('-' * 90)
    for dk in DATASET_KEYS:
        res = all_results[dk]
        sa_g = res['ensemble_results']['SimpleAvg']['gain']
        wa_g = res['ensemble_results']['WeightedAvg']['gain']
        lr_g = res['ensemble_results']['LRStacking']['gain']
        ri_g = res['ensemble_results']['RidgeStacking']['gain']
        fu_g = res['ensemble_results']['FuzzyEnsemble']['gain']
        ad_g = res['ensemble_results']['AdaBoost']['gain']
        print(f'{dk:<20} {sa_g:>+10.4f} {wa_g:>+10.4f} {lr_g:>+10.4f} {ri_g:>+10.4f} {fu_g:>+10.4f} {ad_g:>+10.4f}')

    # 最佳集成方法
    print('\n\nBest Ensemble Method per Dataset:')
    print('{:<20} {:<20} {:>10}'.format('Dataset', 'Best Method', 'Gain'))
    print('-' * 50)
    for dk in DATASET_KEYS:
        res = all_results[dk]
        best_ens = max(res['ensemble_results'].items(), key=lambda x: x[1]['r2'])
        print(f'{dk:<20} {best_ens[0]:<20} {best_ens[1]["gain"]:>+10.4f}')


if __name__ == '__main__':
    run_experiment()