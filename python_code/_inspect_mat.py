"""Inspect .mat file contents to see what variables are available"""
import scipy.io as sio
import numpy as np
import os

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'

# Check a0_Postmonsoon_JAJAPUR.mat
print('='*60)
print('a0_Postmonsoon_JAJAPUR.mat')
print('='*60)
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)
print(f'Keys: {[k for k in mat.keys() if not k.startswith("__")]}')
for k in mat.keys():
    if k.startswith('__'):
        continue
    v = mat[k]
    if hasattr(v, 'dtype'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        if v.dtype.kind in ('i', 'f'):
            print(f'    values={v}')
    else:
        print(f'  {k}: type={type(v)}')

# Check b0_X_GQ.mat
print()
print('='*60)
print('b0_X_GQ.mat (Bootstrap_new)')
print('='*60)
fpath = os.path.join(base, 'Bootstrap_new', 'b0_X_GQ.mat')
mat2 = sio.loadmat(fpath, squeeze_me=True)
print(f'Keys: {[k for k in mat2.keys() if not k.startswith("__")]}')
for k in mat2.keys():
    if k.startswith('__'):
        continue
    v = mat2[k]
    if hasattr(v, 'dtype'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        if v.dtype.kind in ('i', 'f'):
            print(f'    first={v.flatten()[:5]}')
            if v.dtype.kind == 'f':
                print(f'    range=[{v.min():.3f}, {v.max():.3f}]')
    else:
        print(f'  {k}: type={type(v)}')

# Check b0_X_GQ.mat in variable_selection
print()
print('='*60)
print('b0_X_GQ.mat (variable_selection)')
print('='*60)
fpath = os.path.join(base, 'variable_selection', 'b0_X_GQ.mat')
mat3 = sio.loadmat(fpath, squeeze_me=True)
print(f'Keys: {[k for k in mat3.keys() if not k.startswith("__")]}')
for k in mat3.keys():
    if k.startswith('__'):
        continue
    v = mat3[k]
    if hasattr(v, 'dtype'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        if v.dtype.kind in ('i', 'f'):
            print(f'    first={v.flatten()[:5]}')
            if v.dtype.kind == 'f':
                print(f'    range=[{v.min():.3f}, {v.max():.3f}]')
    else:
        print(f'  {k}: type={type(v)}')

# Check PPE_mean.mat
print()
print('='*60)
print('PPE_mean.mat')
print('='*60)
fpath = os.path.join(base, 'PPE_mean.mat')
if os.path.exists(fpath):
    mat4 = sio.loadmat(fpath, squeeze_me=True)
    print(f'Keys: {[k for k in mat4.keys() if not k.startswith("__")]}')
    for k in mat4.keys():
        if k.startswith('__'):
            continue
        v = mat4[k]
        if hasattr(v, 'dtype'):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
            if v.dtype.kind in ('i', 'f'):
                print(f'    first={v.flatten()[:5]}')
                if v.dtype.kind == 'f':
                    print(f'    range=[{v.min():.3f}, {v.max():.3f}]')
        else:
            print(f'  {k}: type={type(v)}')
else:
    print('  NOT FOUND')