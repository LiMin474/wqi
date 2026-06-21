"""
Extract numerical data directly from the __function_workspace__ binary stream.
The MCOS serialization stores table data as cell arrays of column vectors.
"""
import numpy as np
import struct
import os
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')


def extract_numerical_data(fpath):
    """Extract numerical arrays from __function_workspace__"""
    mat = sio.loadmat(fpath, squeeze_me=True)
    
    if '__function_workspace__' not in mat:
        return None
    
    fw = mat['__function_workspace__']
    fw_bytes = fw.tobytes()
    
    print(f'Function workspace size: {len(fw_bytes)} bytes')
    
    # Find "MCOS" marker
    mcos_idx = fw_bytes.find(b'MCOS')
    print(f'MCOS marker at byte {mcos_idx}')
    
    if mcos_idx < 0:
        return None
    
    # Find all occurrences of double-precision data patterns
    # In the MAT file, double arrays are stored as:
    # tag (miDOUBLE=9) + size (4 bytes) + data (8 bytes per element)
    
    # Also look for column names in the binary
    known_columns = ['Station', 'Latitude', 'Longitude', 'pH', 'EC', 'TDS', 'TH',
                     'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3',
                     'Na', 'K', 'Cu', 'Ni', 'GWQI', 'LWQI', 'AWQI', 'RWQI', 'WQI']
    
    column_positions = []
    for col in known_columns:
        idx = fw_bytes.find(col.encode())
        if idx >= 0:
            name_start = max(0, idx - 20)
            context = fw_bytes[name_start:idx+len(col)+10].hex()
            column_positions.append((idx, col, context))
    
    column_positions.sort()
    print(f'\nColumn names found ({len(column_positions)}):')
    for idx, col, ctx in column_positions[:25]:
        print(f'  {idx:6d}: {col}')
    
    # Now search for data blocks after "MCOS"
    # The table data is stored after property metadata
    # Look for groups of double values that could be column data
    
    # Find all miDOUBLE tags (tag=9)
    print(f'\nSearching for miDOUBLE (tag=9) data blocks...')
    double_blocks = []
    for i in range(0, len(fw_bytes) - 12):
        if fw_bytes[i] == 9 and fw_bytes[i+1] == 0 and fw_bytes[i+2] == 0 and fw_bytes[i+3] == 0:
            size = struct.unpack_from('<I', fw_bytes, i+4)[0]
            n_vals = size // 8
            if 70 <= n_vals <= 80:  # 74 samples expected
                values = np.frombuffer(fw_bytes, dtype=np.float64, count=n_vals, offset=i+8)
                double_blocks.append((i, n_vals, values[:5].tolist(), values.min(), values.max()))
    
    if double_blocks:
        print(f'Found {len(double_blocks)} potential 74-element data blocks:')
        for i, n, first5, minv, maxv in double_blocks:
            print(f'  Offset {i}: {n} values, first={first5}, range=[{minv:.2f}, {maxv:.2f}]')
    else:
        print('  No 74-element data blocks found, trying other sizes...')
        # Try wider range
        for i in range(0, len(fw_bytes) - 12):
            if fw_bytes[i] == 9 and fw_bytes[i+1] == 0 and fw_bytes[i+2] == 0 and fw_bytes[i+3] == 0:
                size = struct.unpack_from('<I', fw_bytes, i+4)[0]
                n_vals = size // 8
                if 20 <= n_vals <= 200:
                    values = np.frombuffer(fw_bytes, dtype=np.float64, count=n_vals, offset=i+8)
                    double_blocks.append((i, n_vals, values[:5].tolist(), values.min(), values.max()))
        
        # Filter to likely data blocks (having reasonable water quality values)
        print(f'\nAll double blocks found ({len(double_blocks)}):')
        for i, n, first5, minv, maxv in double_blocks[:40]:
            print(f'  Offset {i:6d}: {n:3d} vals, first={str(first5):40s}, range=[{minv:10.2f}, {maxv:10.2f}]')


if __name__ == '__main__':
    base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
    
    print('='*70)
    print('Decoding: a0_Postmonsoon_JAJAPUR.mat (wqdata + stdwt)')
    print('='*70)
    extract_numerical_data(os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat'))
    
    print('\n' + '='*70)
    print('Decoding: b0_X_GQ.mat (X0 + GQ)')
    print('='*70)
    extract_numerical_data(os.path.join(base, 'Bootstrap_new', 'b0_X_GQ.mat'))