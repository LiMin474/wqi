"""
Extract 2-value miDOUBLE arrays from workspace to get correct stdwt values.
These are [BIS_value, weight] pairs stored for each water quality parameter.
"""
import numpy as np
import os
import scipy.io as sio
import struct
import pandas as pd

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')

mat = sio.loadmat(fpath, squeeze_me=True)
fw = mat['__function_workspace__']
fw_flat = fw[0]
total_len = len(fw_flat)

# Find all 2-value miDOUBLE arrays
two_val_blocks = []
i = 0
while i < total_len - 12:
    if fw_flat[i] == 9 and fw_flat[i+1] == 0 and fw_flat[i+2] == 0 and fw_flat[i+3] == 0:
        size = struct.unpack_from('<I', fw_flat, i+4)[0]
        n_vals = size // 8
        if 0 <= size <= total_len - i - 8 and size % 8 == 0 and n_vals == 2:
            vals = np.frombuffer(fw_flat, dtype=np.float64, count=2, offset=i+8).copy()
            # Get some context bytes before the tag to find column name
            ctx_start = max(0, i - 32)
            ctx = fw_flat[ctx_start:i].tobytes()
            # Try to find ASCII text near this position
            nearby_text = ''
            for j in range(ctx_start, min(total_len, i + 16)):
                if 32 <= fw_flat[j] < 127:
                    nearby_text += chr(fw_flat[j])
                elif nearby_text and len(nearby_text) > 1:
                    break
            two_val_blocks.append((i, vals[0], vals[1], nearby_text))
            i += 8 + size
        elif 0 <= size <= total_len - i - 8 and size % 8 == 0:
            i += 8 + size
        else:
            i += 1
    else:
        i += 1

print(f'Found {len(two_val_blocks)} 2-value arrays:')
for idx, (pos, v0, v1, text) in enumerate(two_val_blocks):
    print(f'  [{idx:2d}] offset {pos:5d} (0x{pos:04x}): [{v0:8.2f}, {v1:4.1f}]  near: ...{text}...')

# The column names in the wqdata table (in order, after removing Lat, Lon, GWQI, LWQI, AWQI, RWQI)
# wqdata column order: Lat, Lon, pH, EC, TDS, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, Na, HCO3, TH, GWQI, LWQI, AWQI, RWQI
# stdwt should match the water quality parameter columns
# But TH and TDS are removed later, so we have 12 params: pH, EC, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, HCO3
# Plus TH and TDS before removal = 14 params

stdwt_param_order = ['pH', 'EC', 'TDS', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'Na', 'HCO3', 'TH']

# Check: we have 14 2-value arrays and 15 params including TH... hmm
# Let me check by mapping the values
print(f'\n\nstdwt_param_order has {len(stdwt_param_order)} params but found {len(two_val_blocks)} arrays')

# Try mapping directly from the 74-value column order  
# wqdata columns (20 after removing TH): Lat, Lon, pH, EC, TDS, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, Na, HCO3, GWQI, LWQI, AWQI, RWQI
# stdwt columns: pH, EC, TDS, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, Na, HCO3
# That's 14 params! And we have 14 2-value arrays! Perfect match.

print('\nMapping to stdwt parameters (14 params):')
stdwt_cols = ['pH', 'EC', 'TDS', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'Na', 'HCO3']
for idx, (pos, v0, v1, text) in enumerate(two_val_blocks):
    if idx < len(stdwt_cols):
        print(f'  {stdwt_cols[idx]:4s}: BIS={v0:.2f}, weight={v1:.1f}')
    else:
        print(f'  [{idx}]: BIS={v0:.2f}, weight={v1:.1f} (extra, might be TH)')