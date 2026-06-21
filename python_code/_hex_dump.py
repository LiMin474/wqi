"""
Look at hex dump of workspace to understand MCOS structure
"""
import scipy.io as sio
import numpy as np
import os

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)
fw = mat['__function_workspace__']
fw_bytes = fw.tobytes()

print(f'__function_workspace__: {len(fw_bytes)} bytes')
print()

# Print full hex dump (every byte)
for i in range(0, len(fw_bytes), 16):
    chunk = fw_bytes[i:i+16]
    hex_str = chunk.hex()
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f'{i:04x}: {hex_str:32s}  {ascii_str}')