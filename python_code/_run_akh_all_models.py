import numpy as np
import os, sys, warnings, time
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from common_codes.a3_Bayesopt_rf_model import a3_Bayesopt_rf_model
from common_codes.a5_Fit_lm_model import a5_Fit_lm_model
from common_codes.a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt

DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')

datasets = {
    'Original (raw Coliforms)': np.load(os.path.join(DATA_DIR, '4_akh_wqi.npz'), allow_pickle=True),
    'Log-transformed Coliforms': np.load(os.path.join(DATA_DIR, '4_akh_wqi_logcoliforms.npz'), allow_pickle=True),
}

print(f"{'Dataset':<30} {'Model':<12} {'R2':>8} {'R2CV':>8} {'Time(s)':>8}")
print('-' * 68)

for dname, data in datasets.items():
    X, y = data['X'], data['y'].ravel()
    
    # 1. Linear Model
    t0 = time.time()
    _, perf = a5_Fit_lm_model(X, y.reshape(-1, 1))
    t = time.time() - t0
    r2 = perf['R2'].values[0]
    r2cv = perf['R2CV'].values[0]
    print(f'{dname:<30} {"LM":<12} {r2:>8.4f} {r2cv:>8.4f} {t:>8.2f}')
    
    # 2. RF (Bayesian-optimized)
    t0 = time.time()
    _, A1 = a3_Bayesopt_rf_model(X, y)
    t = time.time() - t0
    print(f'{dname:<30} {"RF-Bayes":<12} {A1["R2"]:>8.4f} {A1["R2CV"]:>8.4f} {t:>8.2f}')
    
    # 3. ANN-Bayesian
    t0 = time.time()
    _, A1 = a4_Bayesian_fitrnet_opt(X, y)
    t = time.time() - t0
    print(f'{dname:<30} {"ANN-Bayes":<12} {A1["R2"]:>8.4f} {A1["R2CV"]:>8.4f} {t:>8.2f}')
    
    # 4. ANN-SHADE
    t0 = time.time()
    _, A1 = a4_SHADE_fitrnet_opt(X, y)
    t = time.time() - t0
    print(f'{dname:<30} {"ANN-SHADE":<12} {A1["R2"]:>8.4f} {A1["R2CV"]:>8.4f} {t:>8.2f}')
    
    print('-' * 68)