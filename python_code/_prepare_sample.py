import numpy as np
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'datasets', '3_sample_dataset.csv')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'datasets')
SCRIPT_DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')

df = pd.read_csv(CSV_PATH, encoding='latin1')

# Fix conductivity column name
for c in df.columns:
    if 'Conductivity' in str(c):
        df.rename(columns={c: 'Conductivity'}, inplace=True)
        break

feature_cols = ['Alkalinity-total (as CaCO3)', 'Ammonia-Total (as N)', 
                'BOD - 5 days (Total)', 'Chloride', 'Conductivity',
                'Dissolved Oxygen', 'ortho-Phosphate (as P) - unspecified',
                'pH', 'Temperature', 'Total Hardness (as CaCO3)', 'True Colour']

X = df[feature_cols].values.astype(float)
y = df[['CCME_Values']].values.astype(float)

print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'Features: {feature_cols}')
print(f'y range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}')

# Save
for sp in [os.path.join(DATA_DIR, '3_sample_dataset.npz'),
           os.path.join(SCRIPT_DATA_DIR, '3_sample_dataset.npz')]:
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    np.savez_compressed(sp, X=X, y=y, name='Sample Dataset (Lee Cork)',
                       target_name='CCME_Values', n_features=11)
    print(f'Saved: {sp}')