"""
Extract all data from MATLAB .mat files and save as CSVs.
Parses MCOS table data from __function_workspace__ by finding miDOUBLE arrays.
"""
import numpy as np
import os
import scipy.io as sio
import warnings
import pandas as pd
import struct
warnings.filterwarnings('ignore')


def find_midouble_blocks(fpath, target_nvals=[74]):
    """Find all miDOUBLE arrays of target sizes in __function_workspace__"""
    mat = sio.loadmat(fpath, squeeze_me=True)
    if '__function_workspace__' not in mat:
        return {}
    fw = mat['__function_workspace__']
    fw_flat = fw[0]
    total_len = len(fw_flat)
    
    result = {n: [] for n in target_nvals}
    i = 0
    while i < total_len - 12:
        if fw_flat[i] == 9 and fw_flat[i+1] == 0 and fw_flat[i+2] == 0 and fw_flat[i+3] == 0:
            size = struct.unpack_from('<I', fw_flat, i+4)[0]
            # Validate: size must be plausible (non-negative, within remaining bytes, multiple of 8)
            if 0 <= size <= total_len - i - 8 and size % 8 == 0:
                n_vals = size // 8
                if n_vals in target_nvals:
                    vals = np.frombuffer(fw_flat, dtype=np.float64, count=n_vals, offset=i+8).copy()
                    result[n_vals].append((i, vals))
                i += 8 + size
            else:
                i += 1
        else:
            i += 1
    
    return result


base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
save_dir = os.path.join(base, 'python_code')
os.makedirs(save_dir, exist_ok=True)

# =======================================================
# Step 1: Extract GQ directly (it's a regular MATLAB array)
# =======================================================
print('Reading GQ from b0_X_GQ.mat...')
for gq_path in [
    os.path.join(base, 'Bootstrap_new', 'b0_X_GQ.mat'),
    os.path.join(base, 'variable_selection', 'b0_X_GQ.mat')
]:
    if os.path.exists(gq_path):
        mat_gq = sio.loadmat(gq_path, squeeze_me=True)
        if 'GQ' in mat_gq:
            GQ = np.asarray(mat_gq['GQ'], dtype=float).flatten()
            print(f'  GQ: {len(GQ)} values, range=[{GQ.min():.2f}, {GQ.max():.2f}], mean={GQ.mean():.2f}')
            np.savez(os.path.join(save_dir, 'GQ_data.npz'), GQ=GQ)
            break

# =======================================================
# Step 2: Extract 74-value blocks from wqdata file
# =======================================================
print('\nExtracting data from a0_Postmonsoon_JAJAPUR.mat...')
blocks = find_midouble_blocks(os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat'))
wqdata_74 = sorted(blocks[74], key=lambda x: x[0])
print(f'Found {len(wqdata_74)} data columns (74 values each)')

# Expected column names and ranges
# NOTE: After verifying against paper Table 3, our column mapping is:
#   paper_EC = our_TDS (column with [53.4, 3000])
#   paper_CaH = our_MgH (column with [0, 140])
#   paper_MgH = our_Na  (column with [0, 100])
# Our columns named 'EC' and 'CaH' are extra params NOT in the paper's 12-parameter set.
wqdata_cols = ['Latitude', 'Longitude', 'pH', 'EC_raw', 'TDS_as_EC', 'DO', 'F', 'Cl',
               'NO3', 'SO4', 'PO4', 'U', 'CaH_raw', 'MgH_as_CaH', 'Na_as_MgH', 'HCO3',
               'GWQI', 'LWQI', 'AWQI', 'RWQI']

expected_ranges = {
    'Latitude': (20.5, 21.2), 'Longitude': (85.5, 86.8),
    'pH': (5.0, 8.5), 'EC_raw': (20, 1700), 'TDS_as_EC': (50, 3100),
    'DO': (1.0, 8.0), 'F': (0, 2.0), 'Cl': (0, 300),
    'NO3': (4, 55), 'SO4': (2, 110), 'PO4': (0, 4),
    'U': (0, 20), 'CaH_raw': (40, 200), 'MgH_as_CaH': (0, 150),
    'Na_as_MgH': (0, 110), 'HCO3': (20, 400), 'GWQI': (22, 90),
    'LWQI': (22, 90), 'AWQI': (22, 90), 'RWQI': (28, 80)
}

# Match columns by range
wqdata_dict = {}
unused_cols = list(wqdata_cols)
for offset, vals in wqdata_74:
    vmin, vmax = vals.min(), vals.max()
    best_match = None
    for col in unused_cols:
        clo, chi = expected_ranges[col]
        if clo <= vmin <= chi or clo <= vmax <= chi:
            best_match = col
            break
    if best_match:
        wqdata_dict[best_match] = vals
        unused_cols.remove(best_match)
        print(f'  [{len(wqdata_dict)-1:2d}] {best_match:12s}: [{vmin:8.2f}, {vmax:8.2f}]')
    else:
        print(f'  [ ?] ???         : [{vmin:8.2f}, {vmax:8.2f}] (UNMATCHED)')

# Remap columns to match paper's 12-parameter set:
# Paper uses: pH, EC, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, HCO3
paper_12_cols = ['pH', 'EC', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']
remap = {
    'pH': 'pH',
    'TDS_as_EC': 'EC',      # Our TDS column is the paper's EC
    'DO': 'DO',
    'F': 'F',
    'Cl': 'Cl',
    'NO3': 'NO3',
    'SO4': 'SO4',
    'PO4': 'PO4',
    'U': 'U',
    'MgH_as_CaH': 'CaH',    # Our MgH column is the paper's CaH
    'Na_as_MgH': 'MgH',     # Our Na column is the paper's MgH
    'HCO3': 'HCO3'
}
# Extra columns NOT in paper: EC_raw, CaH_raw - drop these

# Build correctly-named DataFrame with paper's 12 params + metadata
paper_wqdata = {}
paper_wqdata['Station'] = [f'S{i+1}' for i in range(74)]
paper_wqdata['Latitude'] = wqdata_dict.get('Latitude', np.zeros(74))
paper_wqdata['Longitude'] = wqdata_dict.get('Longitude', np.zeros(74))

for internal_name, paper_name in remap.items():
    if internal_name in wqdata_dict:
        paper_wqdata[paper_name] = wqdata_dict[internal_name]

# Also keep computed indices if available
for idx_name in ['GWQI', 'LWQI', 'AWQI', 'RWQI']:
    if idx_name in wqdata_dict:
        paper_wqdata[idx_name] = wqdata_dict[idx_name]

wqdata_df = pd.DataFrame(paper_wqdata)
wqdata_df.to_csv(os.path.join(save_dir, 'wqdata.csv'), index=False)
print(f'\nwqdata.csv saved: {wqdata_df.shape}')
print(f'Columns: {list(wqdata_df.columns)}')

# =======================================================
# Step 3: Extract X0 from b0_X_GQ.mat
# Paper uses 12 hydrochemical params: pH, EC, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, HCO3
# =======================================================
print('\nExtracting X0 from variable_selection/b0_X_GQ.mat...')
blocks2 = find_midouble_blocks(os.path.join(base, 'variable_selection', 'b0_X_GQ.mat'))
x0_74 = sorted(blocks2[74], key=lambda x: x[0])
print(f'Found {len(x0_74)} data columns')

x0_col_names = ['pH', 'EC', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']
# NOTE: same column mapping issue as wqdata
# Our blocks might need remapping - let's use stats-based matching
x0_expected_ranges = {
    'pH': (5.0, 8.5), 'EC': (50, 3100), 'DO': (1.0, 8.0),
    'F': (0, 2.0), 'Cl': (0, 300), 'NO3': (4, 55),
    'SO4': (2, 110), 'PO4': (0, 4), 'U': (0, 20),
    'CaH': (0, 150), 'MgH': (0, 110), 'HCO3': (20, 400)
}

x0_dict = {}
unused_x0 = list(x0_col_names)
for offset, vals in x0_74:
    vmin, vmax = vals.min(), vals.max()
    best_match = None
    for col in unused_x0:
        clo, chi = x0_expected_ranges[col]
        if clo <= vmin <= chi or clo <= vmax <= chi:
            best_match = col
            break
    if best_match:
        x0_dict[best_match] = vals
        unused_x0.remove(best_match)
        print(f'  [{len(x0_dict)-1:2d}] {best_match:12s}: [{vmin:8.2f}, {vmax:8.2f}]')
    else:
        print(f'  [ ?] ???         : [{vmin:8.2f}, {vmax:8.2f}] (UNMATCHED)')

if len(x0_dict) >= 12:
    X0 = pd.DataFrame({col: x0_dict[col] for col in x0_col_names})
    X0.to_csv(os.path.join(save_dir, 'X0_data.csv'), index=False)
    print(f'X0_data.csv saved: {X0.shape}')
else:
    # Fallback: extract X0 from wqdata
    print(f'Only {len(x0_dict)}/12 columns matched from b0_X_GQ.mat, using wqdata fallback...')
    if len(wqdata_dict) >= 12:
        x0_final = {}
        for paper_name in x0_col_names:
            if paper_name in wqdata_dict:
                x0_final[paper_name] = wqdata_dict[paper_name]
            else:
                for internal_name, pname in remap.items():
                    if pname == paper_name and internal_name in wqdata_dict:
                        x0_final[paper_name] = wqdata_dict[internal_name]
        if len(x0_final) >= 12:
            X0 = pd.DataFrame({col: x0_final[col] for col in x0_col_names})
            X0.to_csv(os.path.join(save_dir, 'X0_data.csv'), index=False)
            print(f'X0_data.csv saved from wqdata: {X0.shape}')

# =======================================================
# Step 4: BIS Standards (from paper Table 1)
# Paper Table 1: Water quality standard and hydrochemical parameter weight (GWQI model)
# Format: [BIS_value, weight] per parameter
# =======================================================
# Paper Table 1 values (verified):
#   pH=6.5,w=4  EC=2250,w=4  DO=5,w=2  F=1,w=5  Cl=250,w=1
#   NO3=45,w=5  SO4=200,w=3  PO4=10,w=1  U=60,w=5
#   CaH=75,w=2  MgH=30,w=2  HCO3=200,w=3
bis_standards = {
    'pH': [6.5, 4.0], 'EC': [2250.0, 4.0],
    'DO': [5.0, 2.0], 'F': [1.0, 5.0], 'Cl': [250.0, 1.0],
    'NO3': [45.0, 5.0], 'SO4': [200.0, 3.0], 'PO4': [10.0, 1.0],
    'U': [60.0, 5.0], 'CaH': [75.0, 2.0], 'MgH': [30.0, 2.0],
    'HCO3': [200.0, 3.0]
}

# BISd: 2xN array, rows=[BIS_value, weight], cols=parameters
bis_vals = [v[0] for v in bis_standards.values()]
wt_vals = [v[1] for v in bis_standards.values()]
BISd = np.array([bis_vals, wt_vals])  # 2x12

np.savez(os.path.join(save_dir, 'BISd_data.npz'), BISd=BISd)

# stdwt.csv format: 2 rows, columns=parameter names
# Row 0 = BIS values, Row 1 = weights
stdwt_params = list(bis_standards.keys())
stdwt_data = {p: [v[0], v[1]] for p, v in bis_standards.items()}
stdwt_df = pd.DataFrame(stdwt_data, columns=stdwt_params)
stdwt_df.to_csv(os.path.join(save_dir, 'stdwt.csv'), index=False)
print(f'\nstdwt.csv saved: {stdwt_df.shape}')

# =======================================================
# Summary
# =======================================================
print('\n' + '='*60)
print('EXTRACTION SUMMARY')
print('='*60)
for f in ['wqdata.csv', 'X0_data.csv', 'stdwt.csv', 'GQ_data.npz', 'BISd_data.npz']:
    fpath = os.path.join(save_dir, f)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f'  {f}: {size:,} bytes')
    else:
        print(f'  {f}: NOT FOUND')
print('='*60)