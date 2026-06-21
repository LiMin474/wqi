import numpy as np, os, sys, time, warnings
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))
from a3_Bayesopt_rf_model import a3_Bayesopt_rf_model
from a5_Fit_lm_model import a5_Fit_lm_model
from a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt
from a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt

d = np.load(os.path.join(SCRIPT_DIR, 'datasets', '3_sample_dataset.npz'), allow_pickle=True)
X, y_all = d['X'], d['y']
y = y_all.ravel()

print(f"{'Model':<12} {'R2':>8} {'R2CV':>8} {'Time(s)':>8}")
print('-' * 40)

# 1. LM
t0 = time.time()
_, perf = a5_Fit_lm_model(X, y)
t = time.time() - t0
print(f"{'LM':<12} {perf['R2'].values[0]:>8.4f} {perf['R2CV'].values[0]:>8.4f} {t:>8.2f}")

# 2. RF
t0 = time.time()
_, A1 = a3_Bayesopt_rf_model(X, y)
t = time.time() - t0
print(f"{'RF-Bayes':<12} {A1['R2']:>8.4f} {A1['R2CV']:>8.4f} {t:>8.2f}")

# 3. ANN-Bayes
t0 = time.time()
_, A1 = a4_Bayesian_fitrnet_opt(X, y)
t = time.time() - t0
print(f"{'ANN-Bayes':<12} {A1['R2']:>8.4f} {A1['R2CV']:>8.4f} {t:>8.2f}")

# 4. ANN-SHADE
t0 = time.time()
_, A1 = a4_SHADE_fitrnet_opt(X, y)
t = time.time() - t0
print(f"{'ANN-SHADE':<12} {A1['R2']:>8.4f} {A1['R2CV']:>8.4f} {t:>8.2f}")