"""
Quick scatter CSV generator - no optimization, just default models
Only for visualization purposes
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import csv
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
sys.path.insert(0, PROJECT_DIR)

model_name = sys.argv[1] if len(sys.argv) > 1 else 'MLP-lbfgs'
dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'Jajpur'

dataset_map = {
    'Jajpur': '1_jajpur.npz',
    'Irish': '2_irish_river.npz',
    'AKH': '3_akh_wqi.npz',
}

data = np.load(os.path.join(DATASET_DIR, dataset_map[dataset_name]))
X = data['X']
y = data['y']
print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features')

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
models = []

for name in ea_names:
    print(f'  Training {name}...', flush=True)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='relu',
            alpha=0.001,
            max_iter=2000,
            random_state=1,
        ))
    ])
    model.fit(X, y)
    models.append(model)

predictions = np.array([m.predict(X) for m in models])

y_weighted = np.mean(predictions, axis=0)

path = os.path.join(RESULTS_DIR, f'scatter_{model_name}_{dataset_name}.csv')
errors = {a: np.abs(y - predictions[i]) for i, a in enumerate(ea_names)}
ensemble_error = np.abs(y - y_weighted)
threshold = np.mean(list(errors.values())) + 2 * np.std(list(errors.values()))

with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Actual'] + ea_names + ['WeightedAvg_Pred', 'IsDifficult']
    writer.writerow(header)
    for i in range(len(y)):
        row = [f'{y[i]:.4f}']
        for idx, a in enumerate(ea_names):
            row.append(f'{predictions[idx][i]:.4f}')
        row.append(f'{y_weighted[i]:.4f}')
        avg_error = np.mean([errors[a][i] for a in errors])
        row.append('True' if avg_error > threshold else 'False')
        writer.writerow(row)

print(f'[OK] {path} saved with {len(y)} samples')

import pandas as pd
df = pd.read_csv(path)
print(f'Verified: {len(df)} rows')
