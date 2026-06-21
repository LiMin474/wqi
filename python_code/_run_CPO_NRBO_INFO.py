"""
Run CPO, NRBO, INFO on 4 datasets.
These are 3 new algorithms from 2022-2024 literature.
"""
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

from common_codes.a4_CPO_fitrnet_opt import a4_CPO_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt
from common_codes.a4_INFO_fitrnet_opt import a4_INFO_fitrnet_opt


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
    'CPO': a4_CPO_fitrnet_opt,
    'NRBO': a4_NRBO_fitrnet_opt,
    'INFO': a4_INFO_fitrnet_opt,
}

all_results = {}

for key, filename in datasets.items():
    X, y = load_dataset(filename)
    print(f'\n{"="*60}')
    print(f'Dataset: {key}  ({len(y)} samples)')
    print(f'Target: mean={y.mean():.3f}, std={y.std():.3f}')
    print(f'{"="*60}')

    dataset_results = {}

    for method_name, method_func in methods.items():
        print(f'\n--- Running {method_name} ---')
        t0 = time.time()
        try:
            Mdl, A1 = method_func(X, y)
            elapsed = time.time() - t0
            y_pred = Mdl.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            mae = float(mean_absolute_error(y, y_pred))

            dataset_results[method_name] = {
                'R2': float(A1['R2']),
                'R2CV': float(A1['R2CV']),
                'RMSE': rmse,
                'MAE': mae,
                'NumLayers': int(A1.get('NumLayers', -1)),
                'Layer_1': int(A1.get('Layer_1', -1)),
                'Layer_2': int(A1.get('Layer_2', -1)),
                'Activation': str(A1.get('Activation', 'N/A')),
                'Alpha': float(A1.get('Alpha', -1)),
                'Time': elapsed,
            }
            r = dataset_results[method_name]
            print(f'  R2={r["R2"]:.4f}, R2CV={r["R2CV"]:.4f}, RMSE={r["RMSE"]:.4f}, MAE={r["MAE"]:.4f}, Time={r["Time"]:.1f}s')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  {method_name}: FAILED - {str(e)[:200]}')

    all_results[key] = dataset_results

# Save results
save_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'CPO_NRBO_INFO_results.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\n{"="*60}')
print(f'Results saved to: {save_path}')
print(f'{"="*60}')

# Print summary
print('\n=== SUMMARY ===')
for ds in all_results:
    print(f'\n{ds}:')
    for algo in all_results[ds]:
        r = all_results[ds][algo]
        print(f'  {algo}: R2CV={r["R2CV"]:.4f}, RMSE={r["RMSE"]:.4f}, Time={r["Time"]:.1f}s')