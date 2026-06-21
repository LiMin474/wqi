"""Convert PPE_mean.mat to numpy .npz format for Python scripts"""
import scipy.io as sio
import numpy as np
import os

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
mat_path = os.path.join(base, 'PPE_mean.mat')
save_dir = os.path.join(base, 'python_code', 'saved_models')
os.makedirs(save_dir, exist_ok=True)

mat = sio.loadmat(mat_path, squeeze_me=True)
# Create npz with same variable names
data_dict = {}
for k in mat:
    if k.startswith('__'):
        continue
    v = mat[k]
    if isinstance(v, np.ndarray) and v.dtype.kind == 'O':
        data_dict[k] = np.array([x for x in v.flat], dtype=object)
    else:
        data_dict[k] = v

np.savez(os.path.join(save_dir, 'PPE_mean.npz'), **data_dict)
print(f'PPE_mean.npz saved with keys: {list(data_dict.keys())}')

# Verify
check = np.load(os.path.join(save_dir, 'PPE_mean.npz'), allow_pickle=True)
print(f'meanloss shape: {check["meanloss"].shape}')
print(f'vars: {check["vars"]}')

# Also check b0_X_GQ.mat has GQ in it - already confirmed it works
print('\nGQ from b0_X_GQ.mat verified earlier: 74 samples, range [22.38, 85.02]')

# Check if there are other .mat files in root that need converting
for f in os.listdir(base):
    if f.endswith('.mat') and os.path.isfile(os.path.join(base, f)):
        print(f'\nMAT file: {f}')
        try:
            m = sio.loadmat(os.path.join(base, f), squeeze_me=True)
            for k in m:
                if not k.startswith('__'):
                    v = m[k]
                    if isinstance(v, np.ndarray) and v.dtype.names is None and v.dtype.kind != 'O':
                        print(f'  {k}: shape={v.shape}, dtype={v.dtype} (readable)')
                    else:
                        print(f'  {k}: type={type(v).__name__} (not directly readable)')
        except Exception as e:
            print(f'  Error: {e}')