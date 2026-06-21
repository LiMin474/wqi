"""Examine the MCOS table structure - s2 contains table metadata"""
import scipy.io as sio
import numpy as np
import os

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'

# Check both files
for fname, desc in [
    (os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat'), 'wqdata table'),
    (os.path.join(base, 'variable_selection', 'b0_X_GQ.mat'), 'X0 table')
]:
    print(f'='*60)
    print(f'{desc}: {os.path.basename(fname)}')
    print(f'='*60)
    
    mat = sio.loadmat(fname, squeeze_me=True)
    none_var = mat['None']
    
    s2 = none_var['s2']  # table metadata
    print(f's2 type: {type(s2)}')
    
    # s2 is a MatlabOpaque object, get the bytes
    if hasattr(s2, 'item'):
        s2_bytes = s2.item()
        print(f's2 bytes length: {len(s2_bytes)}')
        print(f's2 bytes (first 200): {s2_bytes[:200]}')
        print(f's2 bytes hex: {s2_bytes[:100].hex()}')
        
        # Try to find column names (ASCII strings)
        # Column names in MCOS tables are stored as MATLAB char arrays
        print(f'\ns2 bytes as string (printable):')
        printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in s2_bytes[:500])
        print(printable)
    
    arr = none_var['arr']
    if hasattr(arr, 'item'):
        arr_data = arr.item()
        print(f'\narr type: {type(arr_data)}')
        if isinstance(arr_data, np.ndarray):
            print(f'arr shape: {arr_data.shape}, dtype: {arr_data.dtype}')
            print(f'arr values: {arr_data}')
    
    # Also check s0
    s0 = none_var['s0']
    if hasattr(s0, 'item'):
        s0_bytes = s0.item()
        print(f'\ns0 bytes: {s0_bytes}')
    
    print()

# Now let's also look for any readable float data in the MAT files directly
# by checking the raw file structure
print('\n' + '='*60)
print('Looking for float64 arrays in the raw file...')
print('='*60)

with open(os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat'), 'rb') as f:
    raw = f.read()

print(f'File size: {len(raw)} bytes')

# Search for data element tags (miDOUBLE = 9, miMATRIX = 14)
import struct
tags_found = []
for i in range(0, len(raw) - 8):
    tag = struct.unpack_from('<I', raw, i)[0]
    if tag == 14:  # miMATRIX
        tags_found.append(('miMATRIX', i))
    elif tag == 9:  # miDOUBLE
        tags_found.append(('miDOUBLE', i))

for tag_type, pos in tags_found[:30]:
    print(f'  {tag_type} at offset {pos}')

print(f'\nTotal tags found: {len(tags_found)}')