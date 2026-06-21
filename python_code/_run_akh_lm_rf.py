import numpy as np
import os, sys, warnings, time
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from a3_Bayesopt_rf_model import a3_Bayesopt_rf_model
from a5_Fit_lm_model import a5_Fit_lm_model

DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')

datasets = {
    'Original (raw Coliforms)': np.load(os.path.join(DATA_DIR, '4_akh_wqi.npz'), allow_pickle=True),
    'Log-transformed Coliforms': np.load(os.path.join(DATA_DIR, '4_akh_wqi_logcoliforms.npz'), allow_pickle=True),
}

print(f"{'Dataset':<30} {'Model':<10} {'R2':>8} {'R2CV':>8} {'Time(s)':>8}")
print('-' * 66)

for dname, data in datasets.items():
    X, y = data['X'], data['y'].ravel()
    
    t0 = time.time()
    _, perf = a5_Fit_lm_model(X, y)
    t = time.time() - t0
    print(f'{dname:<30} {"LM":<10} {perf["R2"].values[0]:>8.4f} {perf["R2CV"].values[0]:>8.4f} {t:>8.2f}')
    
    t0 = time.time()
    _, A1 = a3_Bayesopt_rf_model(X, y)
    t = time.time() - t0
    print(f'{dname:<30} {"RF-Bayes":<10} {A1["R2"]:>8.4f} {A1["R2CV"]:>8.4f} {t:>8.2f}')
    print('-' * 66)