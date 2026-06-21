import sys, os, time
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

from data_loader import load_wqdata, load_stdwt
import numpy as np
from sklearn.model_selection import KFold

wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns:
        wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns:
        stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)

from common_codes.a2_GWQI import a2_GWQI
GQ = a2_GWQI(X.copy(), BISd)

from common_codes.a4_DE_feature_selection import SumSqr_DE_FS, decode_params

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cvss = list(kf.split(X))

# Test 1: random params (keep all features)
print("Test 1: random params, all features")
x = np.random.rand(17)
params = decode_params(x)
print(f'  feature_mask: {params[0].sum()} features kept')
start = time.time()
target, output = SumSqr_DE_FS(params, X, GQ, cvss)
elapsed = time.time() - start
print(f'  target={target:.6f}, R2CV={output["R2CV"]:.4f}, time={elapsed:.2f}s')

# Test 2: keep only 3 features
print("\nTest 2: only pH, EC, DO (features 0,1,2)")
x2 = np.zeros(17)
x2[:3] = 1.0  # keep only first 3 features
params2 = decode_params(x2)
print(f'  feature_mask: {params2[0].sum()} features kept')
start = time.time()
target2, output2 = SumSqr_DE_FS(params2, X, GQ, cvss)
elapsed2 = time.time() - start
print(f'  target={target2:.6f}, R2CV={output2["R2CV"]:.4f}, time={elapsed2:.2f}s')

# Test 3: check the original DE speed for comparison
print("\nTest 3: original DE model (same data)")
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

start = time.time()
Mdl = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(4,), activation='relu', solver='lbfgs', alpha=0.001, max_iter=2000, random_state=1, early_stopping=True))
])
Mdl.fit(X, GQ)
y_pred = Mdl.predict(X)
SSEmdl = np.sum((GQ - y_pred)**2)
SST = np.sum((GQ - np.mean(GQ))**2)
R2 = 1 - (SSEmdl / SST)

cv_scores = cross_val_score(Mdl, X, GQ, cv=5, scoring='neg_mean_squared_error')
SSEcv = -cv_scores.sum() * len(GQ) / len(cv_scores)
R2CV = 1 - (SSEcv / SST)
elapsed3 = time.time() - start
print(f'  R2={R2:.4f}, R2CV={R2CV:.4f}, time={elapsed3:.2f}s')

print("\n===== Timing Summary =====")
print(f"DE-FS per evaluation (all features): {elapsed:.2f}s")
print(f"DE-FS per evaluation (3 features):   {elapsed2:.2f}s")
print(f"Original DE per evaluation:          {elapsed3:.2f}s")
print(f"Estimated 68 evals (all):            {elapsed * 68:.0f}s = {elapsed * 68/60:.1f}min")
print(f"Estimated 68 evals (3 feats):        {elapsed2 * 68:.0f}s = {elapsed2 * 68/60:.1f}min")