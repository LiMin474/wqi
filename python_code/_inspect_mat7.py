"""
Parse MCOS table data from __function_workspace__.
MATLAB tables are serialized as MCOS objects with a specific structure.
"""
import scipy.io as sio
import numpy as np
import os
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')

# Load the file
mat = sio.loadmat(fpath, squeeze_me=True)
fw = mat['__function_workspace__']
fw_bytes = fw.tobytes()

print(f'__function_workspace__: {len(fw_bytes)} bytes')

# Print first 1000 bytes in hex
print('\nFirst 1000 bytes hex:')
for i in range(0, min(1000, len(fw_bytes)), 16):
    chunk = fw_bytes[i:i+16]
    hex_str = chunk.hex()
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f'{i:04x}: {hex_str:32s}  {ascii_str}')

# Look for float64 data at 8-byte aligned positions
# by scanning for non-zero, non-denormalized values
print('\n\nSearching for valid water-quality float64 values...')
valid_positions = []
for offset in range(0, len(fw_bytes) - 8, 8):
    try:
        val = struct.unpack_from('<d', fw_bytes, offset)[0]
        if np.isfinite(val) and 0.1 < val < 4000:
            valid_positions.append((offset, val))
    except:
        pass

print(f'Found {len(valid_positions)} valid float64 values')

# Group into contiguous blocks
if valid_positions:
    blocks = []
    current = [valid_positions[0]]
    for i in range(1, len(valid_positions)):
        if valid_positions[i][0] - valid_positions[i-1][0] == 8:
            current.append(valid_positions[i])
        else:
            if len(current) >= 10:
                blocks.append(current)
            current = [valid_positions[i]]
    if len(current) >= 10:
        blocks.append(current)
    
    print(f'\nContiguous data blocks (>=10 values):')
    for bi, block in enumerate(blocks):
        vals = np.array([b[1] for b in block])
        start = block[0][0]
        print(f'  Block {bi}: offset={start}, n={len(vals)}, range=[{vals.min():.3f}, {vals.max():.3f}]')

# Also specifically look for 74-value blocks
print('\n\nLooking specifically for 74-consecutive-float64 blocks:')
found_74 = []
for offset in range(0, len(fw_bytes) - 74*8, 8):
    try:
        vals = np.frombuffer(fw_bytes, dtype=np.float64, count=74, offset=offset)
        if np.all(np.isfinite(vals)):
            vmin, vmax = vals.min(), vals.max()
            if 0 < vmin < 2000 and vmax < 4000:
                found_74.append((offset, vals.copy()))
                print(f'  Offset {offset}: [{vmin:.3f}, {vmax:.3f}], first={vals[:5]}')
    except:
        pass

print(f'\nTotal 74-value blocks found: {len(found_74)}')