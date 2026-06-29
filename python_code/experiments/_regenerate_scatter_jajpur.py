"""
Regenerate scatter_Jajpur.csv with all 74 samples
Run the full experiment for Jajpur only, then save scatter CSV.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import json
import time
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')

def decode_params(x):
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = 5 + int(x[1] * 15)
    layer2 = 5 + int(x[2] * 15)
    activation_idx = int(x[3] * 3)
    activation = ['tanh', 'sigmoid', 'relu'][activation_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha

def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
    MAE = np.mean(np.abs(y_true - y_pred))
    return R2, RMSE, MAE

def evaluate_ann(params, X, y, cvss, max_iter=2000):
    n_layers, layer1, layer2, activation, alpha = params
    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)
    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act_map[activation],
            alpha=alpha,
            max_iter=max_iter,
            random_state=1,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            solver='adam',
            learning_rate_init=0.001
        ))
    ])
    r2cv_list = []
    for train_idx, test_idx in cvss:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        R2CV_fold, _, _ = calc_metrics(y_test, y_pred)
        r2cv_list.append(R2CV_fold)
    R2CV = np.mean(r2cv_list)
    model.fit(X, y)
    y_pred_full = model.predict(X)
    R2, RMSE, MAE = calc_metrics(y, y_pred_full)
    return R2, R2CV, RMSE, MAE, model

def run_algorithm(name, X, y, cvss, max_evals, popsize=10, seed=1, polish=False):
    print(f'  Running {name} (max_evals={max_evals})...', flush=True)
    t0 = time.time()
    bounds = [(0, 1)] * 5

    def objective(x):
        p = decode_params(x)
        _, R2CV, _, _, _ = evaluate_ann(p, X, y, cvss, max_iter=300)
        return 1 - R2CV

    result = differential_evolution(objective, bounds, maxiter=max_evals,
                                    popsize=popsize, seed=seed, workers=1,
                                    updating='deferred', polish=polish)
    best_params = decode_params(result.x)
    R2, R2CV, RMSE, MAE, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)
    time_cost = time.time() - t0
    print(f'    Done: R²={R2:.4f}  R²CV={R2CV:.4f}  RMSE={RMSE:.3f}  MAE={MAE:.3f}  Time={time_cost:.1f}s')
    return {'R2': R2, 'R2CV': R2CV, 'RMSE': RMSE, 'MAE': MAE, 'Time': time_cost,
            'model': model, 'best_params': best_params}

def weighted_avg(predictions, weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(predictions, axis=0, weights=weights)

print('=' * 60)
print('Regenerating scatter_Jajpur.csv (74 samples)')
print('=' * 60)

# Load Jajpur dataset
data = np.load(os.path.join(DATASET_DIR, '1_jajpur.npz'))
X = data['X']
y = data['y']
print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features')

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cvss = list(kf.split(X))

config = {'DE': 5, 'SHADE': 5, 'CMA-ES': 5, 'NRBO': 5, 'BOA': 5, 'HHO-Lite': 5}

ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
seeds = [1, 2, 3, 4, 5, 6]

results = {}
for algo, seed in zip(ea_names, seeds):
    results[algo] = run_algorithm(algo, X, y, cvss, max_evals=config[algo], seed=seed)

# Ensemble predictions
r2cv_scores = [results[a]['R2CV'] for a in ea_names]
models = [results[a]['model'] for a in ea_names]
predictions = np.array([m.predict(X) for m in models])
y_weighted = weighted_avg(predictions, r2cv_scores)

# Save scatter CSV (all 74 samples)
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
        for a in ea_names:
            idx = ea_names.index(a)
            row.append(f'{predictions[idx][i]:.4f}')
        row.append(f'{y_weighted[i]:.4f}')
        avg_error = np.mean([errors[a][i] for a in errors])
        row.append('True' if avg_error > threshold else 'False')
        writer.writerow(row)

print(f'\n[OK] scatter_Jajpur.csv saved with {len(y)} samples to: {path}')

# Verify
import pandas as pd
df = pd.read_csv(path)
print(f'Verified: {len(df)} rows in CSV')