import numpy as np
import os
import sys
import warnings
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

# 六个进化算法 + 贝叶斯对比方法
from common_codes.a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt  # 对比方法
from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt              # DE (1997)
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt        # SHADE (2013)
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt        # CMA-ES (2006)
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt          # NRBO (2024)
from common_codes.a4_BOA_fitrnet_opt import a4_BOA_fitrnet_opt            # BOA (2026)
from common_codes.a4_HHO_Lite_fitrnet_opt import a4_HHO_Lite_fitrnet_opt  # HHO-Lite (2025)
from common_codes.a4_ensemble_stacking import a4_ensemble_stacking


def load_dataset(name):
    """Load .npz dataset from python_code/datasets/"""
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    target_name = str(data['target_name'])
    dataset_name = str(data['name'])
    return X, y, dataset_name, target_name


def run_experiment(dataset_key, X, y, dataset_name, target_name):
    print()
    print('#' * 70)
    print(f'# Dataset: {dataset_name}')
    print(f'# Samples: {len(y)}, Features: {X.shape[1]}, Target: {target_name}')
    print(f'# Target stats: mean={y.mean():.3f}, std={y.std():.3f}')
    print('#' * 70)

    results = {}

    # 六个进化算法 + 贝叶斯对比方法
    methods = {
        'Bayesian': a4_Bayesian_fitrnet_opt,      # 对比方法
        'DE (1997)': a4_DE_fitrnet_opt,           # 差分进化
        'SHADE (2013)': a4_SHADE_fitrnet_opt,     # 成功历史自适应差分进化
        'CMA-ES (2006)': a4_CMAES_fitrnet_opt,    # 协方差矩阵自适应进化策略
        'NRBO (2024)': a4_NRBO_fitrnet_opt,       # 牛顿-拉夫逊基优化器
        'BOA (2026)': a4_BOA_fitrnet_opt,         # 狒狒优化算法
        'HHO-Lite (2025)': a4_HHO_Lite_fitrnet_opt,  # 哈里斯鹰优化精简版
    }

    for method_name, method_func in methods.items():
        print()
        print(f'--- Running {method_name} ---')
        t0 = time.time()
        try:
            Mdl, A1 = method_func(X, y)
            elapsed = time.time() - t0
            y_pred = Mdl.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            results[method_name] = {
                'R2': float(A1.get('R2', np.nan)),
                'R2CV': float(A1.get('R2CV', np.nan)),
                'RMSE': rmse,
                'MAE': mae,
                'NumLayers': int(A1.get('NumLayers', -1)),
                'Layer_1': int(A1.get('Layer_1', -1)),
                'Layer_2': int(A1.get('Layer_2', -1)),
                'Activation': str(A1.get('Activation', 'N/A')),
                'Alpha': float(A1.get('Alpha', np.nan)),
                'Time': elapsed,
            }
            print(f'  {method_name}: R2={results[method_name]["R2"]:.4f}, '
                  f'R2CV={results[method_name]["R2CV"]:.4f}, '
                  f'RMSE={rmse:.4f}, MAE={mae:.4f}, '
                  f'Time={elapsed:.1f}s')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  {method_name}: FAILED - {str(e)[:200]}')
            results[method_name] = {'R2': np.nan, 'R2CV': np.nan, 'Error': str(e)[:200]}

    return results


def print_comparison(dataset_key, results):
    print()
    print(f'{"=" * 90}')
    print(f'  Results Summary: {dataset_key}')
    print(f'{"=" * 90}')
    print(f'{"Method":<12} {"R2":>10} {"R2CV":>10} {"RMSE":>10} {"MAE":>10} {"Layers":>8} {"L1":>5} {"L2":>5} {"Act":>8} {"Alpha":>12} {"Time(s)":>8}')
    print(f'{"-" * 90}')
    for method, r in results.items():
        if 'Error' in r:
            print(f'{method:<12} {"FAILED":>10} {r["Error"][:40]}')
        else:
            print(f'{method:<12} {r["R2"]:>10.4f} {r["R2CV"]:>10.4f} '
                  f'{r["RMSE"]:>10.4f} {r["MAE"]:>10.4f} '
                  f'{r["NumLayers"]:>8} {r["Layer_1"]:>5} {r["Layer_2"]:>5} '
                  f'{r["Activation"]:>8} {r["Alpha"]:>12.6f} {r["Time"]:>8.1f}')


def main():
    # 4 datasets: groundwater → river/lake → surface water → groundwater
    datasets = {
        '1_jajpur': '1_jajpur',
        '2_wqi_dataset': '2_wqi_dataset',
        '3_sample_dataset': '3_sample_dataset',
        '4_akh_wqi': '4_akh_wqi',
    }

    all_results = {}

    for key, filename in datasets.items():
        X, y, dataset_name, target_name = load_dataset(filename)

        # Subsample large datasets (>2000) for speed
        if len(y) > 2000:
            print(f'\n[Note] {dataset_name}: {len(y)} samples, using 2000 random subset')
            np.random.seed(42)
            idx = np.random.choice(len(y), 2000, replace=False)
            X = X[idx]
            y = y[idx]

        results = run_experiment(key, X, y, dataset_name, target_name)
        all_results[key] = results
        print_comparison(key, results)

        # ===== Ensemble: Stacking (DE+SHADE+APSM-jSO+CMA-ES+PSO) =====
        print()
        print(f'--- Ensemble: Stacking (5 algorithms) ---')
        t0_ens = time.time()
        try:
            Mdl_ens, A1_ens = a4_ensemble_stacking(X, y)
            elapsed_ens = time.time() - t0_ens
            y_pred_ens = Mdl_ens.predict(X)
            rmse_ens = np.sqrt(mean_squared_error(y, y_pred_ens))
            mae_ens = mean_absolute_error(y, y_pred_ens)
            results['Ensemble(Stacking)'] = {
                'R2': float(A1_ens.get('R2', np.nan)),
                'R2CV': float(A1_ens.get('R2CV', np.nan)),
                'RMSE': float(rmse_ens),
                'MAE': float(mae_ens),
                'NumLayers': -1,
                'Layer_1': -1,
                'Layer_2': -1,
                'Activation': 'Ensemble',
                'Alpha': -1,
                'Time': elapsed_ens,
            }
            print(f'  Ensemble: R2={results["Ensemble(Stacking)"]["R2"]:.4f}, '
                  f'R2CV={results["Ensemble(Stacking)"]["R2CV"]:.4f}, '
                  f'RMSE={rmse_ens:.4f}, MAE={mae_ens:.4f}, '
                  f'Time={elapsed_ens:.1f}s')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  Ensemble: FAILED - {str(e)[:200]}')

    # ===== Final cross-dataset summary =====
    print()
    print('#' * 80)
    print('# FINAL CROSS-DATASET COMPARISON (4 datasets × 10 algorithms)')
    print('#' * 80)
    print(f'{"Dataset":<20} {"Method":<12} {"R2":>10} {"R2CV":>10} {"RMSE":>10} {"MAE":>10}')
    print(f'{"-" * 70}')
    for key, results in all_results.items():
        for method, r in results.items():
            r2 = f'{r["R2"]:.4f}' if 'R2' in r and not np.isnan(r['R2']) else 'N/A'
            r2cv = f'{r["R2CV"]:.4f}' if 'R2CV' in r and not np.isnan(r['R2CV']) else 'N/A'
            rmse = f'{r["RMSE"]:.4f}' if 'RMSE' in r and not np.isnan(r['RMSE']) else 'N/A'
            mae = f'{r["MAE"]:.4f}' if 'MAE' in r and not np.isnan(r['MAE']) else 'N/A'
            print(f'{key:<20} {method:<12} {r2:>10} {r2cv:>10} {rmse:>10} {mae:>10}')

    # ===== Per-dataset best algorithm =====
    print()
    print('#' * 80)
    print('# BEST ALGORITHM PER DATASET (by R2CV)')
    print('#' * 80)
    for key, results in all_results.items():
        best_method = None
        best_r2cv = -1
        for method, r in results.items():
            if 'R2CV' in r and not np.isnan(r['R2CV']) and r['R2CV'] > best_r2cv:
                best_r2cv = r['R2CV']
                best_method = method
        if best_method:
            print(f'{key:<20} Best: {best_method:<12} R2CV={best_r2cv:.4f}')
        else:
            print(f'{key:<20} All methods FAILED')

    # ===== Save results =====
    save_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'all_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()