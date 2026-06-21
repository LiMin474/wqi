"""
修复数据泄露的集成实验：训练集/测试集分离 + Out-of-Fold预测
==============================================================
"""

import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_APSM_jSO_fitrnet_opt import a4_APSM_jSO_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    return X, y


datasets = {
    '1_jajpur': '1_jajpur',
    '2_wqi_dataset': '2_wqi_dataset',
    '3_sample_dataset': '3_sample_dataset',
    '4_akh_wqi': '4_akh_wqi',
}

methods = {
    'DE': a4_DE_fitrnet_opt,
    'SHADE': a4_SHADE_fitrnet_opt,
    'APSM-jSO': a4_APSM_jSO_fitrnet_opt,
    'CMA-ES': a4_CMAES_fitrnet_opt,
    'NRBO': a4_NRBO_fitrnet_opt,
}


def simple_avg(predictions):
    """简单平均"""
    return np.mean(predictions, axis=0)


def weighted_avg(predictions, r2cv_scores):
    """按R²CV加权平均"""
    weights = np.array([max(s, 0) for s in r2cv_scores])
    weights = weights / weights.sum()
    return np.average(predictions, axis=0, weights=weights)


def stacking_lr(predictions_train, y_train, predictions_test):
    """线性回归Stacking"""
    lr = LinearRegression()
    lr.fit(predictions_train.T, y_train)
    return lr.predict(predictions_test.T), lr.coef_


def stacking_ridge(predictions_train, y_train, predictions_test):
    """岭回归Stacking"""
    ridge = Ridge(alpha=1.0)
    ridge.fit(predictions_train.T, y_train)
    return ridge.predict(predictions_test.T), ridge.coef_


all_results = {}
RANDOM_STATE = 42

for ds_key, ds_name in datasets.items():
    print(f'\n{"="*70}')
    print(f'Processing dataset: {ds_key}')
    print(f'{"="*70}')
    
    X, y = load_dataset(ds_name)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_samples = len(y)
    
    # ✅ 正确：划分训练集/测试集 (80/20)
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f'n_train={len(train_idx)}, n_test={len(test_idx)}')
    
    # Step 1: 在训练集上运行5个基算法
    base_models_train = {}
    oof_preds_train = []  # Out-of-Fold预测（用于训练元学习器）
    r2cv_scores = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for method_name, method_func in methods.items():
        print(f'\n--- Training {method_name} ---')
        
        # ✅ 正确：使用5-fold CV生成OOF预测
        n_train = len(y_train)
        oof_pred = np.zeros(n_train)
        
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            Mdl, A1 = method_func(X_tr, y_tr)
            oof_pred[va_idx] = Mdl.predict(X_va)
        
        # 在完整训练集上训练最终模型
        Mdl_full, A1_full = method_func(X_train, y_train)
        
        base_models_train[method_name] = {
            'Model': Mdl_full,
            'R2CV': float(r2_score(y_train, oof_pred))
        }
        oof_preds_train.append(oof_pred)
        r2cv_scores.append(base_models_train[method_name]['R2CV'])
        
        print(f'  R2CV={base_models_train[method_name]["R2CV"]:.4f}')
    
    oof_preds_train = np.array(oof_preds_train)
    
    # Step 2: 在测试集上获取各算法预测
    test_preds = []
    for method_name in methods.keys():
        Mdl = base_models_train[method_name]['Model']
        test_preds.append(Mdl.predict(X_test))
    test_preds = np.array(test_preds)
    
    # Step 3: 评估各集成方法（在测试集上）
    ensemble_results = {}
    
    # SimpleAvg
    print('\n--- Evaluating SimpleAvg (on test set) ---')
    y_ensemble = simple_avg(test_preds)
    ensemble_results['SimpleAvg'] = {
        'R2': float(r2_score(y_test, y_ensemble)),
        'R2CV': float(r2_score(y_test, y_ensemble)),  # 测试集R2作为最终评估
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_ensemble))),
        'MAE': float(mean_absolute_error(y_test, y_ensemble)),
        'Method': 'SimpleAvg'
    }
    print(f'  R2={ensemble_results["SimpleAvg"]["R2"]:.4f}')
    
    # WeightedAvg
    print('\n--- Evaluating WeightedAvg (on test set) ---')
    y_ensemble = weighted_avg(test_preds, r2cv_scores)
    weights = np.array([max(s, 0) for s in r2cv_scores])
    weights = weights / weights.sum()
    ensemble_results['WeightedAvg'] = {
        'R2': float(r2_score(y_test, y_ensemble)),
        'R2CV': float(r2_score(y_test, y_ensemble)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_ensemble))),
        'MAE': float(mean_absolute_error(y_test, y_ensemble)),
        'Method': 'WeightedAvg',
        'Weights': {name: float(w) for name, w in zip(methods.keys(), weights)}
    }
    print(f'  R2={ensemble_results["WeightedAvg"]["R2"]:.4f}')
    
    # LR Stacking ✅ 正确：用OOF训练，用测试集评估
    print('\n--- Evaluating LRStacking (on test set) ---')
    y_ensemble_train, _ = stacking_lr(oof_preds_train, y_train, oof_preds_train)
    y_ensemble_test, coefs = stacking_lr(oof_preds_train, y_train, test_preds)
    
    if len(coefs.shape) > 1:
        coefs = coefs.flatten()
    ensemble_results['LRStacking'] = {
        'R2': float(r2_score(y_test, y_ensemble_test)),
        'R2CV': float(r2_score(y_test, y_ensemble_test)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_ensemble_test))),
        'MAE': float(mean_absolute_error(y_test, y_ensemble_test)),
        'Method': 'LRStacking',
        'Coefs': {name: float(c) for name, c in zip(methods.keys(), coefs)}
    }
    print(f'  R2={ensemble_results["LRStacking"]["R2"]:.4f}')
    
    # Ridge Stacking
    print('\n--- Evaluating RidgeStacking (on test set) ---')
    y_ensemble_train, _ = stacking_ridge(oof_preds_train, y_train, oof_preds_train)
    y_ensemble_test, coefs = stacking_ridge(oof_preds_train, y_train, test_preds)
    
    if len(coefs.shape) > 1:
        coefs = coefs.flatten()
    ensemble_results['RidgeStacking'] = {
        'R2': float(r2_score(y_test, y_ensemble_test)),
        'R2CV': float(r2_score(y_test, y_ensemble_test)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_ensemble_test))),
        'MAE': float(mean_absolute_error(y_test, y_ensemble_test)),
        'Method': 'RidgeStacking',
        'Coefs': {name: float(c) for name, c in zip(methods.keys(), coefs)}
    }
    print(f'  R2={ensemble_results["RidgeStacking"]["R2"]:.4f}')
    
    # Step 4: 记录单算法最佳结果（在测试集上）
    best_single_name = None
    best_single_r2 = -np.inf
    best_single_time = 0
    
    for method_name in methods.keys():
        Mdl = base_models_train[method_name]['Model']
        y_pred = Mdl.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_single_r2:
            best_single_r2 = r2
            best_single_name = method_name
    
    ensemble_results['BestSingle'] = {
        'Algorithm': best_single_name,
        'R2CV': float(best_single_r2),
        'Time': 0
    }
    
    all_results[ds_key] = ensemble_results
    
    # Print summary
    print(f'\n{"-"*70}')
    print(f'Summary for {ds_key}:')
    print(f'  Best Single: {best_single_name} (R2={best_single_r2:.4f})')
    for method_name in ['SimpleAvg', 'WeightedAvg', 'LRStacking', 'RidgeStacking']:
        er = ensemble_results[method_name]
        gain = er['R2CV'] - best_single_r2
        print(f'  {method_name}: R2={er["R2CV"]:.4f} (Gain={gain:+.4f})')

# Save results
save_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'new_ensemble_results_fixed.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{"="*70}')
print(f'Results saved to: {save_path}')
print(f'{"="*70}')

# Final summary
print('\n=== FINAL SUMMARY (Test Set Evaluation) ===')
for ds in all_results:
    print(f'\n{ds}:')
    best_single = all_results[ds]['BestSingle']['R2CV']
    for method in ['SimpleAvg', 'WeightedAvg', 'LRStacking', 'RidgeStacking']:
        r2cv = all_results[ds][method]['R2CV']
        gain = r2cv - best_single
        print(f'  {method}: R2={r2cv:.4f} (vs single {best_single:.4f}, gain={gain:+.4f})')
