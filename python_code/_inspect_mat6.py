"""
Scan the raw .mat file for all float64 data arrays.
Since the workspace contains the actual table data, let's scan it.
"""
import scipy.io as sio
import numpy as np
import os
import struct

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')

# Load the mat and get workspace
mat = sio.loadmat(fpath, squeeze_me=True, mat_dtype=True)
fw = mat['__function_workspace__']
print(f'__function_workspace__ shape: {fw.shape}, size: {fw.size} bytes')
print(f'First 50 bytes: {fw.flatten()[:50].tobytes().hex()}')

# The workspace is uint8 raw data. Let's scan it for valid float64 arrays
# We need to find where float64 data starts
fw_bytes = fw.flatten()

# Let's manually find data patterns - scan for valid float64 at each 8-byte boundary
print('\nScanning for valid 74-element float64 arrays...')
candidates = []
for offset in range(0, len(fw_bytes) - 74*8, 8):
    try:
        vals = np.frombuffer(fw_bytes, dtype=np.float64, count=74, offset=offset)
        if np.all(np.isfinite(vals)):
            vmin, vmax = vals.min(), vals.max()
            # Valid data should have reasonable range
            if 0 <= vmin <= 5000 and vmax > 0:
                candidates.append((offset, vmin, vmax, vals[:5].copy()))
    except:
        pass

print(f'Found {len(candidates)} candidate 74-element arrays')
for offset, vmin, vmax, first5 in candidates[:30]:
    print(f'  Offset {offset:6d}: [{vmin:10.3f}, {vmax:10.3f}], first={first5}')

# Also try to find all float64 values that look like wqdata params
# Check for blocks at 8-byte aligned positions
print('\n\nChecking all float64 values at each 8-byte boundary (first 1000 values):')
all_vals = []
for offset in range(0, min(len(fw_bytes) - 8, 20000), 8):
    val = struct.unpack_from('<d', fw_bytes, offset)[0]
    if np.isfinite(val) and 0 < val < 5000:
        all_vals.append((offset, val))

print(f'Found {len(all_vals)} valid float64 values')
# Group by nearby offsets to find contiguous blocks
if all_vals:
    blocks = []
    current_block = [all_vals[0]]
    for i in range(1, len(all_vals)):
        if all_vals[i][0] - all_vals[i-1][0] <= 8:
            current_block.append(all_vals[i])
        else:
            if len(current_block) >= 10:
                blocks.append(current_block)
            current_block = [all_vals[i]]
    if len(current_block) >= 10:
        blocks.append(current_block)
    
    print(f'Found {len(blocks)} contiguous data blocks:')
    for i, block in enumerate(blocks):
        start = block[0][0]
        end = block[-1][0]
        n_vals = len(block)
        vals_arr = np.array([b[1] for b in block])
        print(f'  Block {i}: offset={start}, n_vals={n_vals}, range=[{vals_arr.min():.3f}, {vals_arr.max():.3f}]')