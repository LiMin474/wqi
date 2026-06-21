"""
Debug: check why find_midouble_blocks finds 0 columns in wqdata file
"""
import numpy as np
import os
import scipy.io as sio
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')

mat = sio.loadmat(fpath, squeeze_me=True)
fw = mat['__function_workspace__']
print(f'fw shape: {fw.shape}, dtype: {fw.dtype}')
fw_flat = fw[0]
print(f'fw_flat len: {len(fw_flat)}')

# Count all tag=9 occurrences
tag9_positions = []
for i in range(len(fw_flat) - 12):
    if fw_flat[i] == 9 and fw_flat[i+1] == 0 and fw_flat[i+2] == 0 and fw_flat[i+3] == 0:
        size = struct.unpack_from('<I', fw_flat, i+4)[0]
        n_vals = size // 8
        tag9_positions.append((i, size, n_vals))

print(f'\nTotal miDOUBLE tags: {len(tag9_positions)}')
print(f'n_vals distribution:')
from collections import Counter
c = Counter([x[2] for x in tag9_positions])
for n, cnt in sorted(c.items()):
    print(f'  {n:4d} vals: {cnt}')

print(f'\n74-value arrays:')
for pos, size, n in tag9_positions:
    if n == 74:
        vals = np.frombuffer(fw_flat, dtype=np.float64, count=74, offset=pos+8)
        print(f'  offset {pos} (0x{pos:04x}): range=[{vals.min():.3f}, {vals.max():.3f}]')

# Now test the exact function logic but without the skip-ahead loop
print(f'\n\nTesting with skip-ahead logic...')
i = 0
count_74 = 0
while i < len(fw_flat) - 12:
    if fw_flat[i] == 9 and fw_flat[i+1] == 0 and fw_flat[i+2] == 0 and fw_flat[i+3] == 0:
        size = struct.unpack_from('<I', fw_flat, i+4)[0]
        n_vals = size // 8
        if n_vals == 74:
            count_74 += 1
            if count_74 <= 3:
                vals = np.frombuffer(fw_flat, dtype=np.float64, count=74, offset=i+8).copy()
                print(f'  Found at {i}: size={size}, first={vals[:3]}')
        i += 8 + size
    else:
        i += 1

print(f'Count with skip-ahead: {count_74}')