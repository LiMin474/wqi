"""Deep inspect the table structure in a0_Postmonsoon_JAJAPUR.mat"""
import scipy.io as sio
import numpy as np
import os

base = r'd:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
fpath = os.path.join(base, 'a0_Postmonsoon_JAJAPUR.mat')
mat = sio.loadmat(fpath, squeeze_me=True)

none_var = mat['None']
print(f'none_var shape: {none_var.shape}')
print(f'none_var dtype: {none_var.dtype}')
print()

# Examine the structured array
for field in none_var.dtype.names:
    val = none_var[field]
    print(f'Field: {field}')
    print(f'  type: {type(val)}')
    if hasattr(val, 'shape'):
        print(f'  shape: {val.shape}')
        if hasattr(val, 'dtype'):
            print(f'  dtype: {val.dtype}')
    if isinstance(val, np.ndarray):
        if val.dtype.kind == 'O':
            print(f'  object array with {val.size} elements')
            for i, item in enumerate(val.flat):
                print(f'    [{i}]: type={type(item).__name__}', end='')
                if isinstance(item, np.ndarray):
                    print(f', shape={item.shape}, dtype={item.dtype}', end='')
                    if item.dtype.kind in ('i', 'f'):
                        print(f', vals={item}', end='')
                elif isinstance(item, str):
                    print(f', val="{item}"', end='')
                print()
        elif val.dtype.kind in ('i', 'f'):
            print(f'  values: {val}')
    elif isinstance(val, str):
        print(f'  value: "{val}"')

# Check __function_workspace__ size
print()
print('='*60)
print('__function_workspace__ info:')
fw = none_var['arr']
print(f'  arr shape: {fw.shape}')
print(f'  arr dtype: {fw.dtype}')
print(f'  arr size: {fw.size} bytes')

# Also check variable_selection/b0_X_GQ.mat for X0 data
print()
print('='*60)
print('variable_selection/b0_X_GQ.mat - checking for X0')
fpath2 = os.path.join(base, 'variable_selection', 'b0_X_GQ.mat')
mat2 = sio.loadmat(fpath2, squeeze_me=True)

none2 = mat2['None']
print(f'none_var shape: {none2.shape}')
print(f'none_var dtype: {none2.dtype}')
for field in none2.dtype.names:
    val = none2[field]
    print(f'Field: {field}')
    print(f'  type: {type(val)}')
    if hasattr(val, 'shape'):
        print(f'  shape: {val.shape}')
    if isinstance(val, np.ndarray) and val.dtype.kind == 'O':
        for i, item in enumerate(val.flat):
            print(f'    [{i}]: type={type(item).__name__}', end='')
            if isinstance(item, np.ndarray):
                print(f', shape={item.shape}, dtype={item.dtype}')
            elif isinstance(item, str):
                print(f', val="{item}"')
            else:
                print(f', val={item}')