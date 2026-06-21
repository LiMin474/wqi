import numpy as np
import os, sys, warnings, time
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt

# Load original and transformed versions
DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')
data_orig = np.load(os.path.join(DATA_DIR, '4_akh_wqi.npz'), allow_pickle=True)
data_log = np.load(os.path.join(DATA_DIR, '4_akh_wqi_logcoliforms.npz'), allow_pickle=True)

datasets = {
    'Original (raw Coliforms)': (data_orig['X'], data_orig['y']),
    'Log-transformed Coliforms': (data_log['X'], data_log['y']),
}

for name, (X, y) in datasets.items():
    print('=' * 70)
    print(f'# {name}')
    print(f'# Samples: {len(y)}, Features: {X.shape[1]}')
    print('=' * 70)
    
    for algo_name, algo_func in [('Bayesian', a4_Bayesian_fitrnet_opt), ('SHADE', a4_SHADE_fitrnet_opt)]:
        print(f'\n--- {algo_name} ---')
        t0 = time.time()
        try:
            Mdl, A1 = algo_func(X, y)
            elapsed = time.time() - t0
            print(f'  {algo_name}: R2={A1["R2"]:.4f}, R2CV={A1["R2CV"]:.4f}, Time={elapsed:.1f}s')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  {algo_name}: FAILED - {str(e)[:200]}')
    print()