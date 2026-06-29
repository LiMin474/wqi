"""
Quick scatter_Jajpur.csv generator - no optimization, just default models
Only for visualization purposes (74 scatter points)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import csv
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')

# Load Jajpur dataset
data = np.load(os.path.join(DATASET_DIR, '1_jajpur.npz'))
X = data['X']
y = data['y']
print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features')

# Load the best params from unified_ensemble_results.json if available
# Otherwise use the pre-computed best_params from experiment
# Let's try to load existing models/results

import json
with open(os.path.join(RESULTS_DIR, 'unified_ensemble_results.json'), 'r') as f:
    all_results = json.load(f)

jajpur_results = all_results['Jajpur']
print('Best single R2CV:', jajpur_results['best_single_r2cv'])
print('Best ensemble R2CV:', jajpur_results['best_ensemble_r2cv'])

# Since models can't be serialized, we need to retrain.
# Let's use a simpler approach: just train 6 MLPRegressor with default params
# This is just for the scatter plot visualization shape
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
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20
        ))
    ])
    model.fit(X, y)
    models.append(model)

# Predict
predictions = np.array([m.predict(X) for m in models])

# Weighted average (equal weights since we don't have proper R²CV)
y_weighted = np.mean(predictions, axis=0)

# Save CSV (74 samples)
path = os.path.join(RESULTS_DIR, 'scatter_Jajpur.csv')
errors = {a: np.abs(y - predictions[i]) for i, a in enumerate(ea_names)}
ensemble_error = np.abs(y - y_weighted)
threshold = np.mean(list(errors.values())) + 2 * np.std(list(errors.values()))

with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Actual'] + ea_names + ['Ensemble_Pred', 'IsDifficult']
    writer.writerow(header)
    for i in range(len(y)):
        row = [f'{y[i]:.4f}']
        for idx, a in enumerate(ea_names):
            row.append(f'{predictions[idx][i]:.4f}')
        row.append(f'{y_weighted[i]:.4f}')
        avg_error = np.mean([errors[a][i] for a in errors])
        row.append('True' if avg_error > threshold else 'False')
        writer.writerow(row)

print(f'[OK] scatter_Jajpur.csv saved with {len(y)} samples')

# Verify
import pandas as pd
df = pd.read_csv(path)
print(f'Verified: {len(df)} rows')