import sys, os, time
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

from data_loader import load_wqdata, load_stdwt
import numpy as np
from sklearn.model_selection import KFold

wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns: wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns: stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)

from common_codes.a2_GWQI import a2_GWQI
GQ = a2_GWQI(X.copy(), BISd)
print(f'Data: {X.shape}, GQ: {GQ.shape}', flush=True)

from common_codes.a4_DE_feature_selection import decode_params, make_pipe, SumSqr_DE_FS, VAR_NAMES

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cvss = list(kf.split(X))

# Quick test: 1 training with 12 features, max_iter=500
print('\nTest 1: single evaluation with all 12 features', flush=True)
x = np.ones(17)  # all features, approx middle values
x[12] = 0.66  # relu
x[13] = 0.3   # 1 layer
x[14] = 0.5   # 6 neurons
params = decode_params(x)
t0 = time.time()
target, output = SumSqr_DE_FS(params, X, GQ, cvss, max_iter=500)
print(f'  R2CV={output["R2CV"]:.4f}, time={time.time()-t0:.2f}s', flush=True)

# Test 2: sparse features, 2 layers
print('\nTest 2: 3 features (pH, EC, DO), 2 layers', flush=True)
x2 = np.zeros(17)
x2[:3] = 1.0
x2[12] = 0.33  # sigmoid
x2[13] = 0.7   # 2 layers
x2[14] = 0.3   # 4 neurons
x2[15] = 0.8   # 8 neurons
params2 = decode_params(x2)
t0 = time.time()
target2, output2 = SumSqr_DE_FS(params2, X, GQ, cvss, max_iter=500)
print(f'  R2CV={output2["R2CV"]:.4f}, time={time.time()-t0:.2f}s', flush=True)

# Test 3: Run just the first 5 evaluations of DE (initial pop)
print('\nTest 3: Running 5 DE iterations manually', flush=True)
n_params = 17
pop = np.random.rand(5, n_params)
for i in range(5):
    t0 = time.time()
    params_i = decode_params(pop[i])
    target_i, output_i = SumSqr_DE_FS(params_i, X, GQ, cvss, max_iter=500)
    kept = [VAR_NAMES[j] for j in range(12) if params_i[0][j]]
    print(f'  [{i}] n_feat={output_i["n_features"]}, R2CV={output_i["R2CV"]:.4f}, '
          f'kept={kept}, time={time.time()-t0:.2f}s', flush=True)

print('\nAll tests passed! No hanging issues.', flush=True)