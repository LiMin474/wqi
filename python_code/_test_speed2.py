import sys, os, time
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

import numpy as np
from sklearn.model_selection import KFold

from data_loader import load_wqdata, load_stdwt
from common_codes.a2_GWQI import a2_GWQI
from common_codes.a4_DE_feature_selection import SumSqr_DE_FS, decode_params

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

# Test fast eval (max_iter=500) vs full eval (max_iter=2000)
x = np.random.rand(17)
params = decode_params(x)
print(f'Test: {params[0].sum()} features, {"1 layer" if params[2]==1 else "2 layers"}')

start = time.time()
t1, o1 = SumSqr_DE_FS(params, X, GQ, cvss, max_iter=500)
t_fast = time.time() - start
print(f'  max_iter=500: R2CV={o1["R2CV"]:.4f}, time={t_fast:.2f}s')

start = time.time()
t2, o2 = SumSqr_DE_FS(params, X, GQ, cvss, max_iter=2000)
t_full = time.time() - start
print(f'  max_iter=2000: R2CV={o2["R2CV"]:.4f}, time={t_full:.2f}s')

print(f'\nSpeed ratio: {t_full/t_fast:.1f}x faster with max_iter=500')
print(f'Estimated 170 evals with max_iter=500: {t_fast * 170 / 60:.1f} min')