import sys, os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, os.path.join(_BASE_DIR, 'common_codes'))

from data_loader import load_wqdata, load_stdwt
from a2_GWQI import a2_GWQI
from a4_DE_fitrnet_opt import a4_DE_fitrnet_opt

wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns:
        wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns:
        stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X0 = wqdata.iloc[:, 3:15]
X = X0.values.astype(float)
GQ = a2_GWQI(X, BISd)
print(f'Data loaded: {X.shape[0]} samples')
print('Testing DE...')
Mdl, Opt = a4_DE_fitrnet_opt(X, GQ)
print(f'DE done: R2={Opt["R2"]:.4f}, R2CV={Opt["R2CV"]:.4f}')
print(f'Best: L1={Opt["Layer_1"]}, L2={Opt["Layer_2"]}, Act={Opt["Activation"]}, Alpha={Opt["Alpha"]:.6f}')