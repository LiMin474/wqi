"""
Run ensemble experiments with new algorithm combination:
DE + SHADE + APSM-jSO + CMA-ES + NRBO

Compare multiple ensemble methods:
1. SimpleAvg - 等权平均
2. WeightedAvg - 按R²CV加权平均 (论文核心)
3. RidgeStacking - 岭回归Stacking
4. LRStacking - 线性回归Stacking
"""
import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

for ds_key, ds_name in datasets.items():
    print(f'\n{"="*70}')
    print(f'Processing dataset: {ds_key}')
    print(f'{"="*70}')
    
    X, y = load_dataset(ds_name)
    n_samples = len(y)
    
    # Step 1: 运行5个基算法，获取模型和预测
    base_models = {}
    base_preds = []
    r2cv_scores = []
    
    for method_name, method_func in methods.items():
        print(f'\n--- Training {method_name} ---')
        t0 = time.time()
        Mdl, A1 = method_func(X, y)
        elapsed = time.time() - t0
        
        y_pred = Mdl.predict(X)
        r2cv = float(A1['R2CV'])
        
        base_models[method_name] = {
            'R2CV': r2cv,
            'Time': elapsed,
            'Model': Mdl
        }
        base_preds.append(y_pred)
        r2cv_scores.append(r2cv)
        
        print(f'  R2CV={r2cv:.4f}, Time={elapsed:.1f}s')
    
    base_preds = np.array(base_preds)
    
    # Step 2: 应用不同集成方法
    ensemble_results = {}
    
    # SimpleAvg
    print('\n--- Evaluating SimpleAvg ---')
    y_ensemble = simple_avg(base_preds)
    ensemble_results['SimpleAvg'] = {
        'R2': float(r2_score(y, y_ensemble)),
        'R2CV': float(r2_score(y, y_ensemble)),  # 用整体数据评估
        'RMSE': float(np.sqrt(mean_squared_error(y, y_ensemble))),
        'MAE': float(mean_absolute_error(y, y_ensemble)),
        'Method': 'SimpleAvg'
    }
    print(f'  R2={ensemble_results["SimpleAvg"]["R2"]:.4f}, R2CV={ensemble_results["SimpleAvg"]["R2CV"]:.4f}')
    
    # WeightedAvg
    print('\n--- Evaluating WeightedAvg ---')
    y_ensemble = weighted_avg(base_preds, r2cv_scores)
    weights = np.array([max(s, 0) for s in r2cv_scores])
    weights = weights / weights.sum()
    ensemble_results['WeightedAvg'] = {
        'R2': float(r2_score(y, y_ensemble)),
        'R2CV': float(r2_score(y, y_ensemble)),
        'RMSE': float(np.sqrt(mean_squared_error(y, y_ensemble))),
        'MAE': float(mean_absolute_error(y, y_ensemble)),
        'Method': 'WeightedAvg',
        'Weights': {name: float(w) for name, w in zip(methods.keys(), weights)}
    }
    print(f'  R2={ensemble_results["WeightedAvg"]["R2"]:.4f}, R2CV={ensemble_results["WeightedAvg"]["R2CV"]:.4f}')
    print(f'  Weights: {ensemble_results["WeightedAvg"]["Weights"]}')
    
    # LR Stacking
    print('\n--- Evaluating LRStacking ---')
    y_ensemble, coefs = stacking_lr(base_preds, y, base_preds)
    # 处理可能的二维数组
    if len(coefs.shape) > 1:
        coefs = coefs.flatten()
    ensemble_results['LRStacking'] = {
        'R2': float(r2_score(y, y_ensemble)),
        'R2CV': float(r2_score(y, y_ensemble)),
        'RMSE': float(np.sqrt(mean_squared_error(y, y_ensemble))),
        'MAE': float(mean_absolute_error(y, y_ensemble)),
        'Method': 'LRStacking',
        'Coefs': {name: float(c) for name, c in zip(methods.keys(), coefs)}
    }
    print(f'  R2={ensemble_results["LRStacking"]["R2"]:.4f}, R2CV={ensemble_results["LRStacking"]["R2CV"]:.4f}')
    
    # Ridge Stacking
    print('\n--- Evaluating RidgeStacking ---')
    y_ensemble, coefs = stacking_ridge(base_preds, y, base_preds)
    # 处理可能的二维数组
    if len(coefs.shape) > 1:
        coefs = coefs.flatten()
    ensemble_results['RidgeStacking'] = {
        'R2': float(r2_score(y, y_ensemble)),
        'R2CV': float(r2_score(y, y_ensemble)),
        'RMSE': float(np.sqrt(mean_squared_error(y, y_ensemble))),
        'MAE': float(mean_absolute_error(y, y_ensemble)),
        'Method': 'RidgeStacking',
        'Coefs': {name: float(c) for name, c in zip(methods.keys(), coefs)}
    }
    print(f'  R2={ensemble_results["RidgeStacking"]["R2"]:.4f}, R2CV={ensemble_results["RidgeStacking"]["R2CV"]:.4f}')
    
    # Step 3: 记录单算法最佳结果
    best_single = max(base_models.items(), key=lambda x: x[1]['R2CV'])
    ensemble_results['BestSingle'] = {
        'Algorithm': best_single[0],
        'R2CV': best_single[1]['R2CV'],
        'Time': best_single[1]['Time']
    }
    
    all_results[ds_key] = ensemble_results
    
    # Print summary for this dataset
    print(f'\n{"-"*70}')
    print(f'Summary for {ds_key}:')
    print(f'  Best Single: {best_single[0]} (R2CV={best_single[1]["R2CV"]:.4f})')
    for method_name in ['SimpleAvg', 'WeightedAvg', 'LRStacking', 'RidgeStacking']:
        er = ensemble_results[method_name]
        gain = er['R2CV'] - best_single[1]['R2CV']
        print(f'  {method_name}: R2CV={er["R2CV"]:.4f} (Gain={gain:+.4f})')

# Save results
save_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'new_ensemble_results.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{"="*70}')
print(f'Results saved to: {save_path}')
print(f'{"="*70}')

# Final summary
print('\n=== FINAL SUMMARY ===')
for ds in all_results:
    print(f'\n{ds}:')
    best_single = all_results[ds]['BestSingle']['R2CV']
    for method in ['SimpleAvg', 'WeightedAvg', 'LRStacking', 'RidgeStacking']:
        r2cv = all_results[ds][method]['R2CV']
        gain = r2cv - best_single
        print(f'  {method}: R2CV={r2cv:.4f} (vs single {best_single:.4f}, gain={gain:+.4f})')