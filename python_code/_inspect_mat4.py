"""Better search for data in function workspace"""
import scipy.io as sio
import numpy as np
import os
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)

fw = mat['__function_workspace__']
print(f'__function_workspace__ shape: {fw.shape}, dtype: {fw.dtype}')
print(f'First 200 bytes hex:')
print(fw.flatten()[:200].tobytes().hex())
print()

# Search for tag 9 (miDOUBLE) in various forms
fw_bytes = fw.flatten().tobytes()

# Look for the miDOUBLE tag 9 as a 32-bit LE integer
print('Searching for miDOUBLE (tag=9, 4-byte LE)...')
found = []
for i in range(0, len(fw_bytes) - 12):
    tag = struct.unpack_from('<I', fw_bytes, i)[0]
    if tag == 9:  # miDOUBLE
        size = struct.unpack_from('<I', fw_bytes, i+4)[0]
        n_vals = size // 8
        found.append((i, size, n_vals))
        if len(found) <= 30:
            print(f'  Offset {i}: size={size}, n_vals={n_vals}')
            if n_vals <= 100 and n_vals > 0:
                vals = np.frombuffer(fw_bytes, dtype=np.float64, count=n_vals, offset=i+8)
                print(f'    vals: first={vals[:3]}, range=[{vals.min():.4f}, {vals.max():.4f}]')

print(f'\nTotal miDOUBLE blocks: {len(found)}')

# Also search for miMATRIX tag (14) which might wrap data
print('\nSearching for miMATRIX (tag=14) ...')
for i in range(0, min(500, len(fw_bytes) - 12)):
    tag = struct.unpack_from('<I', fw_bytes, i)[0]
    if tag == 14:
        print(f'  Offset {i}: tag=14 (miMATRIX)')
        print(f'    surrounding bytes: {fw_bytes[max(0,i-4):i+20].hex()}')

# Search for any float64 data pattern
print('\nSearching for any plausible float64 arrays (8-byte aligned, reasonable values)...')
for i in range(0, len(fw_bytes) - 80, 8):  # step by 8 for alignment
    if i % 8 != 0:
        continue
    # Check first few values
    try:
        vals = np.frombuffer(fw_bytes, dtype=np.float64, count=10, offset=i)
        if np.all(np.isfinite(vals)):
            vmin, vmax = vals.min(), vals.max()
            # Look for data in range typical for water quality
            if 0 < vmin < 2000 and 0 < vmax < 4000:
                # Check if next 74 values also reasonable
                vals74 = np.frombuffer(fw_bytes, dtype=np.float64, count=74, offset=i)
                if np.all(np.isfinite(vals74)):
                    print(f'  Offset {i}: [{vmin:.2f}, {vmax:.2f}], first={vals[:5]}, 74-range=[{vals74.min():.2f}, {vals74.max():.2f}]')
    except:
        pass