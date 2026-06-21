"""
Search for miDOUBLE arrays of all sizes in __function_workspace__
"""
import scipy.io as sio
import numpy as np
import os
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)
fw = mat['__function_workspace__']

# fw is shape (1, 31592), get flat view
fw_flat = fw[0]
print(f'__function_workspace__: {len(fw_flat)} bytes')

# Search for miDOUBLE (tag=9) at all positions
from collections import Counter
size_counts = Counter()
tag9_positions = []

for i in range(len(fw_flat) - 8):
    if fw_flat[i] == 9 and fw_flat[i+1] == 0 and fw_flat[i+2] == 0 and fw_flat[i+3] == 0:
        size = struct.unpack_from('<I', fw_flat, i+4)[0]
        n_vals = size // 8
        tag9_positions.append((i, size, n_vals))
        if n_vals <= 200:  # only count smaller arrays
            size_counts[n_vals] += 1

print(f'\nFound {len(tag9_positions)} miDOUBLE tag (9) occurrences')
print(f'\nNon-trivial array sizes (≤200 values):')
for n_vals, count in sorted(size_counts.items()):
    print(f'  {n_vals:4d} values ({n_vals*8:5d} bytes): {count} occurrences')

# Show positions for 74-value arrays
print(f'\n74-value array positions:')
for pos, size, n_vals in tag9_positions:
    if n_vals == 74:
        vals = np.frombuffer(fw_flat, dtype=np.float64, count=74, offset=pos+8)
        print(f'  Offset {pos:5d} (0x{pos:04x}): range=[{vals.min():10.3f}, {vals.max():10.3f}], first={vals[:5]}')

# Also check for 20-value and 22-value arrays (might be stdwt/BIS)
print(f'\nOther interesting array positions:')
for pos, size, n_vals in tag9_positions:
    if n_vals in [1, 2, 12, 20, 21, 22, 24]:
        if n_vals <= 24:
            vals = np.frombuffer(fw_flat, dtype=np.float64, count=n_vals, offset=pos+8)
            print(f'  Offset {pos:5d} (0x{pos:04x}): {n_vals:2d} vals = {vals}')

# Now check what's right before each 74-value array
print(f'\nBytes before each 74-value array:')
for pos, size, n_vals in tag9_positions:
    if n_vals == 74:
        before = fw_flat[max(0,pos-16):pos]
        print(f'  Offset {pos:5d} (0x{pos:04x}): preceding 16 bytes = {before.tobytes().hex()}')
        # Check if there's a column name nearby
        nearby = fw_flat[pos:pos+16]
        print(f'    following 16 bytes = {nearby.tobytes().hex()}')