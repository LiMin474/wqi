import sys, os, time
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

import numpy as np
from sklearn.model_selection import KFold

from data_loader import load_wqdata, load_stdwt
from common_codes.a2_GWQI import a2_GWQI
from common_codes.a4_DE_feature_selection import SumSqr_DE_FS, decode_params, VAR_NAMES

wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns: wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns: stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)
GQ = a2_GWQI(X.copy(), BISd)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cvss = list(kf.split(X))

# Simple random search: just 50 random evaluations
print(f'Running random search for feature selection (50 evaluations)...')
print(f'Data: {X.shape[0]} samples, {X.shape[1]} features')

np.random.seed(1)
best_target = float('inf')
best_params = None
best_output = None
best_x = None

times = []
start_all = time.time()

for i in range(50):
    t0 = time.time()
    x = np.random.rand(17)
    params = decode_params(x)
    target, output = SumSqr_DE_FS(params, X, GQ, cvss, max_iter=500)
    elapsed = time.time() - t0
    times.append(elapsed)

    if target < best_target:
        best_target = target
        best_params = params
        best_output = output
        best_x = x
        kept = [VAR_NAMES[j] for j in range(12) if params[0][j]]
        print(f'  [{i+1:2d}] R2CV={output["R2CV"]:.4f} | n={output["n_features"]} | {kept} | {elapsed:.2f}s')

total_time = time.time() - start_all
print(f'\nRandom search complete: {total_time:.1f}s total, {np.mean(times):.2f}s avg per eval')

# Final retrain with max_iter=2000
print(f'\nRetraining best model with max_iter=2000...')
t0 = time.time()
target, output = SumSqr_DE_FS(best_params, X, GQ, cvss, max_iter=2000)
elapsed = time.time() - t0
print(f'Retrain complete: {elapsed:.1f}s')

feature_mask = output['feature_mask']
kept_vars = [VAR_NAMES[i] for i in range(12) if feature_mask[i]]
dropped_vars = [VAR_NAMES[i] for i in range(12) if not feature_mask[i]]

print('\n' + '='*60)
print('DE Feature Selection Results (Random Search)')
print('='*60)
print(f'Kept ({len(kept_vars)}):  {kept_vars}')
print(f'Dropped ({len(dropped_vars)}): {dropped_vars}' if dropped_vars else 'Dropped: none')
print(f'Architecture: {best_params[2]} layer(s), L1={best_params[3]}, L2={best_params[4]}')
print(f'Activation: {best_params[1]}, Alpha={best_params[5]:.6f}')
print(f'R2  = {output["R2"]:.4f}')
print(f'R2CV = {output["R2CV"]:.4f}')
print('='*60)