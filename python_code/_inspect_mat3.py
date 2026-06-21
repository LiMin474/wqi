"""Extract data properly from function workspace"""
import scipy.io as sio
import numpy as np
import os
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'

# Let's examine the function_workspace blob in detail
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)

# Get the __function_workspace__ from the raw MATLAB file
# The 'None' variable has arr which points to workspace data
# But the raw __function_workspace__ is a global variable

if '__function_workspace__' in mat:
    fw = mat['__function_workspace__']
    print(f'__function_workspace__ type: {type(fw)}')
    if hasattr(fw, 'shape'):
        print(f'  shape: {fw.shape}')
        print(f'  dtype: {fw.dtype}')
        if fw.dtype == np.uint8:
            print(f'  size: {fw.size} bytes')
            # search for miDOUBLE tags (9)
            print()
            print('Searching for miDOUBLE (tag=9) blocks...')
            count_74 = 0
            count_1 = 0
            others = []
            for i in range(0, len(fw) - 12):
                if fw[i] == 9 and fw[i+1] == 0 and fw[i+2] == 0 and fw[i+3] == 0:
                    size = struct.unpack_from('<I', fw, i+4)[0]
                    n_vals = size // 8
                    if n_vals == 74:
                        count_74 += 1
                        vals = np.frombuffer(fw, dtype=np.float64, count=n_vals, offset=i+8)
                        print(f'  Block at {i}: 74 vals, range=[{vals.min():.2f}, {vals.max():.2f}], first={vals[:3]}')
                    elif n_vals == 1:
                        count_1 += 1
                        vals = np.frombuffer(fw, dtype=np.float64, count=1, offset=i+8)
                        print(f'  Block at {i}: 1 val = {vals[0]:.4f}')
                    else:
                        if n_vals < 200:  # skip very large blocks
                            others.append((i, n_vals))
            
            print(f'\nSummary: {count_74} blocks of 74 values, {count_1} blocks of 1 value')
            if others:
                print(f'Other blocks: {others[:10]}')
else:
    print('No __function_workspace__ in mat directly')
    # Check if it's in the raw file
    print('Checking raw file...')
    with open(fpath, 'rb') as f:
        raw = f.read()
    
    # Search for 'function_workspace' in raw bytes
    idx = raw.find(b'__function_workspace__')
    print(f'__function_workspace__ found at byte {idx}')
    
    # Search for miDOUBLE tags
    print('\nSearching for miDOUBLE (tag=9) blocks in raw file...')
    count_74 = 0
    count_1 = 0
    for i in range(0, len(raw) - 12):
        if raw[i] == 9 and raw[i+1] == 0 and raw[i+2] == 0 and raw[i+3] == 0:
            size = struct.unpack_from('<I', raw, i+4)[0]
            n_vals = size // 8
            if n_vals == 74:
                count_74 += 1
                if count_74 <= 25:
                    vals_raw = raw[i+8:i+8+min(size, 40)]
                    print(f'  Block at {i}: 74 vals, first bytes={vals_raw[:24].hex()}')
            elif n_vals == 1:
                count_1 += 1
                if count_1 <= 25:
                    vals_raw = raw[i+8:i+16]
                    val = struct.unpack('<d', vals_raw)[0]
                    print(f'  Block at {i}: 1 val = {val:.6f}')
    
    print(f'\nSummary: {count_74} blocks of 74 values, {count_1} blocks of 1 value')
    
    # Let's also look at what other data sizes exist
    size_counts = {}
    for i in range(0, len(raw) - 12):
        if raw[i] == 9 and raw[i+1] == 0 and raw[i+2] == 0 and raw[i+3] == 0:
            size = struct.unpack_from('<I', raw, i+4)[0]
            n_vals = size // 8
            if n_vals not in size_counts:
                size_counts[n_vals] = 0
            size_counts[n_vals] += 1
    
    print('\nBlock size distribution:')
    for n_vals, cnt in sorted(size_counts.items()):
        if cnt > 0:
            print(f'  {n_vals:6d} vals: {cnt} blocks')