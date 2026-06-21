import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

from data_loader import load_wqdata, load_stdwt
import numpy as np

wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns: wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns: stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)

from common_codes.a2_GWQI import a2_GWQI
GQ = a2_GWQI(X.copy(), BISd)
print(f'Data: {X.shape}, GQ: {GQ.shape}')

from common_codes.a4_DE_feature_selection import a4_DE_feature_selection
Modells_ann_fs, Opttable_ann_fs = a4_DE_feature_selection(X, GQ)
print('\n===== Final DE Feature Selection Results =====')
print(f'Kept vars:   {Opttable_ann_fs["KeptVars"]}')
print(f'Dropped vars:{Opttable_ann_fs["DroppedVars"]}')
print(f'R2={Opttable_ann_fs["R2"]:.4f}, R2CV={Opttable_ann_fs["R2CV"]:.4f}')