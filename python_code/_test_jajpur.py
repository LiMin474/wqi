import numpy as np
import sys, os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

data = np.load(os.path.join(SCRIPT_DIR, 'datasets', '1_jajpur.npz'), allow_pickle=True)
X = data['X']
y = data['y']
print(f'Jajpur: {len(y)} samples, {X.shape[1]} features')
print(f'Train shape: {X.shape}, target shape: {y.shape}')

from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
Mdl, A1 = a4_SHADE_fitrnet_opt(X, y)
print(f'SHADE: R2={A1["R2"]:.4f}, R2CV={A1["R2CV"]:.4f}')
print('Done!')