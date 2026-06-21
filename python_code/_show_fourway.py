"""Quick four-way comparison: load previous models + run DE-FS only"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))

import numpy as np
import joblib

from data_loader import load_wqdata, load_stdwt
from common_codes.a2_GWQI import a2_GWQI
from common_codes.a4_DE_feature_selection import a4_DE_feature_selection

# Load data
wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns: wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns: stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)
GQ = a2_GWQI(X.copy(), BISd)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(save_dir, exist_ok=True)

# Load previous optimization results
opt_files = {
    'Bayesian': 'Opttable_ann.npz',
    'DE (L2)': 'Opttable_ann_de.npz',
    'DE-Reg': 'Opttable_ann_reg.npz',
}
results = {}
for name, fname in opt_files.items():
    fpath = os.path.join(save_dir, fname)
    if os.path.exists(fpath):
        data = np.load(fpath, allow_pickle=True)
        results[name] = {k: data[k].item() if data[k].ndim == 0 else data[k]
                         for k in data.keys()}
        print(f'Loaded {name}: R2={results[name]["R2"]:.4f}, R2CV={results[name]["R2CV"]:.4f}')
    else:
        print(f'  {fpath} not found, skipping')

# Run DE Feature Selection
print('\n' + '='*60)
print('  Running DE Feature Selection (simultaneous FS + HP tuning)')
print('='*60)
Modells_ann_fs, Opttable_ann_fs = a4_DE_feature_selection(X, GQ)

results['DE-FS'] = Opttable_ann_fs

# Save DE-FS results
joblib.dump(Modells_ann_fs, os.path.join(save_dir, 'Modells_ann_fs.pkl'))
np.savez(os.path.join(save_dir, 'Opttable_ann_fs.npz'),
         **{k: np.array(v) if not isinstance(v, (list, np.ndarray)) else np.array(v, dtype=object)
            for k, v in Opttable_ann_fs.items()})

# Four-way comparison
print('\n' + '='*70)
print('              FOUR-WAY COMPARISON')
print('='*70)
print(f'{"Metric":<25} {"Bayesian":<14} {"DE (L2)":<14} {"DE-Reg":<14} {"DE-FS":<14}')
print(f'{"Features":<25} {"12":<14} {"12":<14} {"12":<14} {str(Opttable_ann_fs["N_Features"]):<14}')
print('-'*70)
for metric in ['R2', 'R2CV']:
    vals = [f'{results[m][metric]:.4f}' for m in ['Bayesian', 'DE (L2)', 'DE-Reg', 'DE-FS']]
    print(f'{metric:<25} {vals[0]:<14} {vals[1]:<14} {vals[2]:<14} {vals[3]:<14}')
print('-'*70)
print(f'\nDE-FS Kept features ({Opttable_ann_fs["N_Features"]}):')
print(f'  {Opttable_ann_fs["KeptVars"]}')
print(f'DE-FS Dropped features ({len(Opttable_ann_fs["DroppedVars"])}):')
print(f'  {Opttable_ann_fs["DroppedVars"]}')
print(f'DE-FS Architecture: {Opttable_ann_fs["NumLayers"]} layer(s), '
      f'L1={Opttable_ann_fs["Layer_1"]}, L2={Opttable_ann_fs["Layer_2"]}')
print(f'DE-FS Activation: {Opttable_ann_fs["Activation"]}, '
      f'Alpha={Opttable_ann_fs["Alpha"]:.6f}')

# Find best R2CV model
best_model = max(results, key=lambda m: results[m]['R2CV'])
print(f'\n>>> Best model by R2CV: {best_model} (R2CV={results[best_model]["R2CV"]:.4f})')
print('='*70)