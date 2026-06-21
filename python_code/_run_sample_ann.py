import numpy as np, os, sys, time, warnings
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))
from a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt
from a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt

d = np.load(os.path.join(SCRIPT_DIR, 'datasets', '3_sample_dataset.npz'), allow_pickle=True)
X, y = d['X'], d['y'].ravel()

print('--- ANN-Bayesian ---', flush=True)
t0 = time.time()
_, A1 = a4_Bayesian_fitrnet_opt(X, y)
t = time.time() - t0
print(f'ANN-Bayes: R2={A1["R2"]:.4f}, R2CV={A1["R2CV"]:.4f}, Time={t:.1f}s', flush=True)

print('--- ANN-SHADE ---', flush=True)
t0 = time.time()
_, A1 = a4_SHADE_fitrnet_opt(X, y)
t = time.time() - t0
print(f'ANN-SHADE: R2={A1["R2"]:.4f}, R2CV={A1["R2CV"]:.4f}, Time={t:.1f}s', flush=True)