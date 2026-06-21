import numpy as np
import os
import sys
import warnings
import time
import json
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt

print('=' * 60)
print('Multi-Dataset Experiment: SHADE vs CMA-ES')
print('=' * 60)

datasets = [
    ('1_jajpur', 'Jajpur Groundwater (Original)'),
    ('indian_river', 'Indian River Water Quality'),
    ('uci_water_quality', 'UCI Water Quality (pH)'),
]

all_results = {}

for key, desc in datasets:
    print()
    print('#' * 60)
    print(f'# Dataset: {desc}')
    print('#' * 60)

    data = np.load(os.path.join(SCRIPT_DIR, 'datasets', f'{key}.npz'), allow_pickle=True)
    X = data['X']
    y = data['y']

    if len(y) > 1500:
        print(f'  Using 1500 random subset (from {len(y)} samples)')
        np.random.seed(42)
        idx = np.random.choice(len(y), 1500, replace=False)
        X = X[idx]
        y = y[idx]

    print(f'  Samples: {len(y)}, Features: {X.shape[1]}')
    print(f'  Target: mean={y.mean():.3f}, std={y.std():.3f}')

    results = {}

    for method_name, method_func in [('SHADE', a4_SHADE_fitrnet_opt),
                                      ('CMA-ES', a4_CMAES_fitrnet_opt)]:
        print(f'\n  --- {method_name} ---')
        t0 = time.time()
        try:
            Mdl, A1 = method_func(X, y)
            elapsed = time.time() - t0
            results[method_name] = {
                'R2': float(A1.get('R2', np.nan)),
                'R2CV': float(A1.get('R2CV', np.nan)),
                'Layers': f'{A1.get("NumLayers","?")}-{A1.get("Layer_1","?")}-{A1.get("Layer_2","?")}',
                'Activation': str(A1.get('Activation', 'N/A')),
                'Alpha': float(A1.get('Alpha', np.nan)),
                'Time': elapsed,
            }
            print(f'  {method_name}: R2={results[method_name]["R2"]:.4f}, '
                  f'R2CV={results[method_name]["R2CV"]:.4f}, '
                  f'Time={elapsed:.1f}s')
        except Exception as e:
            print(f'  {method_name}: FAILED - {e}')
            results[method_name] = {'R2': np.nan, 'R2CV': np.nan, 'Error': str(e)}

    all_results[key] = results

    print(f'\n  {"="*50}')
    print(f'  {"Method":<12} {"R2":>10} {"R2CV":>10} {"Layers":>12} {"Act":>8} {"Alpha":>10} {"Time":>8}')
    print(f'  {"-"*50}')
    for method, r in results.items():
        if 'Error' in r:
            print(f'  {method:<12} {"FAILED":>10}')
        else:
            print(f'  {method:<12} {r["R2"]:>10.4f} {r["R2CV"]:>10.4f} '
                  f'{r["Layers"]:>12} {r["Activation"]:>8} {r["Alpha"]:>10.6f} {r["Time"]:>8.1f}s')

print()
print('#' * 60)
print('# FINAL COMPARISON')
print('#' * 60)
print(f'{"Dataset":<22} {"Method":<10} {"R2":>10} {"R2CV":>10}')
print(f'{"-"*54}')
for key, results in all_results.items():
    for method, r in results.items():
        r2 = f'{r["R2"]:.4f}' if not np.isnan(r['R2']) else 'N/A'
        r2cv = f'{r["R2CV"]:.4f}' if not np.isnan(r['R2CV']) else 'N/A'
        print(f'{key:<22} {method:<10} {r2:>10} {r2cv:>10}')

save_path = os.path.join(SCRIPT_DIR, 'datasets', 'all_results.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\nResults saved to: {save_path}')
print('Done!')