"""
Phase 2: Feature Selection experiment.
Runs DE-FS and CMA-ES-FS on each dataset, comparing selected-feature results
with the Phase 1 baseline (all features).
"""
import numpy as np
import os
import sys
import warnings
import time
import json
import copy
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_pipe(hidden_layer_sizes, activation, alpha, max_iter):
    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act_map[activation],
            solver='lbfgs',
            alpha=alpha,
            max_iter=max_iter,
            random_state=1,
            n_iter_no_change=10,
            tol=1e-4,
        ))
    ])


def decode_params_generic(x, n_features):
    """Decode [0,1]^n vector to (feature_mask, activation, n_layers, layer1, layer2, alpha)"""
    feature_mask = np.array([xi > 0.5 for xi in x[:n_features]])
    if np.sum(feature_mask) == 0:
        feature_mask[0] = True  # ensure at least 1 feature

    idx = n_features
    act_idx = min(int(x[idx] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    idx += 1

    n_layers = 1 if x[idx] < 0.5 else 2
    idx += 1

    layer1 = int(round(2 + x[idx] * 8))
    layer1 = max(2, min(10, layer1))
    idx += 1

    layer2 = int(round(2 + x[idx] * 8))
    layer2 = max(2, min(10, layer2))
    idx += 1

    alpha = 10.0 ** (-6.0 + x[idx] * 5.0)
    return feature_mask, activation, n_layers, layer1, layer2, alpha


def eval_with_fs(params_decoded, XX, YY, cvss, max_iter):
    """Evaluate one set of parameters with feature selection."""
    feature_mask, activation, n_layers, layer1, layer2, alpha = params_decoded
    n_features_selected = np.sum(feature_mask)

    if n_features_selected == 0:
        return 1.0, {'R2': 0.0, 'R2CV': 0.0, 'n_features': 0}

    XX_sub = XX[:, feature_mask]
    y_all = YY.ravel()
    SST = np.sum((y_all - np.mean(y_all))**2)

    hidden_layer_sizes = (layer1,) if n_layers == 1 else (layer1, layer2)

    # 5-fold CV
    all_preds = np.zeros_like(y_all)
    for train_idx, val_idx in cvss:
        X_tr, X_va = XX_sub[train_idx], XX_sub[val_idx]
        y_tr = y_all[train_idx]
        pipe = make_pipe(hidden_layer_sizes, activation, alpha, max_iter)
        pipe.fit(X_tr, y_tr)
        all_preds[val_idx] = pipe.predict(X_va).ravel()

    SSEcv = np.sum((y_all - all_preds)**2)
    R2CV = 1 - (SSEcv / SST)

    # Final retrain
    final_pipe = make_pipe(hidden_layer_sizes, activation, alpha, max_iter)
    final_pipe.fit(XX_sub, y_all)
    y_pred = final_pipe.predict(XX_sub).ravel()
    SSEmdl = np.sum((y_all - y_pred)**2)
    R2 = 1 - (SSEmdl / SST)

    output = {
        'R2': R2, 'R2CV': R2CV, 'Mdl': final_pipe,
        'feature_mask': feature_mask,
        'n_features': int(n_features_selected)
    }
    return 1 - R2CV, output


def run_DE_fs(XX, YY, cvss, feature_names, max_fast=200, max_final=2000, popsize_mult=1):
    """Differential Evolution with feature selection (generic n_features)."""
    n_features = XX.shape[1]
    n_params = n_features + 5
    pop_size = max(int(n_params * popsize_mult), 10)
    max_gen = max(int(60 / pop_size), 3)
    total_evals = pop_size + max_gen * pop_size
    F = 0.8
    Cr = 0.7

    print(f'  DE-FS ({n_features}+5={n_params}-dim, pop={pop_size}, gen={max_gen}, ~{total_evals} evals)...')

    pop = np.random.rand(pop_size, n_params)
    fitness = np.full(pop_size, np.inf)
    best_f = np.inf
    best_r2cv = 0.0
    best_nfeat = n_features
    eval_count = 0

    def clip_pop(x):
        return np.clip(x, 0.0, 1.0)

    for i in range(pop_size):
        params = decode_params_generic(pop[i], n_features)
        target, output = eval_with_fs(params, XX, YY, cvss, max_fast)
        fitness[i] = target
        eval_count += 1
        if target < best_f:
            best_f = target
            best_r2cv = output['R2CV']
            best_nfeat = output['n_features']
            kept = [feature_names[j] for j in range(n_features) if params[0][j]]
            print(f'    Init {i+1:2d}: R2CV={output["R2CV"]:.4f} | n={output["n_features"]} | {kept[:5]}...')

    for gen in range(max_gen):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = clip_pop(a + F * (b - c))
            trial = np.where(np.random.rand(n_params) < Cr, mutant, pop[i])
            j_rand = np.random.randint(n_params)
            trial[j_rand] = mutant[j_rand]
            trial = clip_pop(trial)

            params_t = decode_params_generic(trial, n_features)
            target_t, output_t = eval_with_fs(params_t, XX, YY, cvss, max_fast)
            eval_count += 1

            if target_t < fitness[i]:
                pop[i] = trial
                fitness[i] = target_t

            if target_t < best_f:
                best_f = target_t
                best_r2cv = output_t['R2CV']
                best_nfeat = output_t['n_features']
                kept = [feature_names[j] for j in range(n_features) if params_t[0][j]]
                print(f'    Gen {gen+1} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | n={output_t["n_features"]} | {kept[:5]}...')

        best_idx = np.argmin(fitness)
        print(f'    >>> Gen {gen+1}: best R2CV={1-fitness[best_idx]:.4f} (n={output.get("n_features","?")})')

    # Final retrain with best
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params_generic(best_x, n_features)
    target, output = eval_with_fs(best_params, XX, YY, cvss, max_final)

    kept_vars = [feature_names[j] for j in range(n_features) if output['feature_mask'][j]]
    dropped_vars = [feature_names[j] for j in range(n_features) if not output['feature_mask'][j]]

    A1 = {
        'Method': 'DE-FS',
        'FeatureMask': output['feature_mask'].tolist(),
        'KeptVars': kept_vars,
        'DroppedVars': dropped_vars,
        'N_Features': output['n_features'],
        'Activation': best_params[1],
        'NumLayers': best_params[2],
        'Layer_1': best_params[3],
        'Layer_2': best_params[4],
        'Alpha': best_params[5],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'Evals': eval_count,
    }

    print(f'    DE-FS done: R2={output["R2"]:.4f}, R2CV={output["R2CV"]:.4f}, kept={output["n_features"]}/{n_features}')
    print(f'    Kept: {kept_vars}')
    if dropped_vars:
        print(f'    Dropped: {dropped_vars}')

    return A1


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    target_name = str(data['target_name'])
    dataset_name = str(data['name'])
    return X, y, dataset_name, target_name


def get_feature_names(n_features):
    """Generate generic feature names if not Jajpur."""
    return [f'F{i+1}' for i in range(n_features)]


def main():
    # Load Phase 1 baseline results for comparison
    baseline_path = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_results.json')
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print('Loaded Phase 1 baseline results.')
    else:
        print('No baseline results found, running feature selection standalone.')

    datasets = {
        '1_jajpur': '1_jajpur',
        '2_wqi_dataset': '2_wqi_dataset',
        '4_akh_wqi': '4_akh_wqi',
    }

    all_results = {}

    for key, filename in datasets.items():
        X, y, dataset_name, target_name = load_dataset(filename)

        print()
        print('#' * 70)
        print(f'# Phase 2: Feature Selection on {dataset_name}')
        print(f'# Samples: {len(y)}, Features: {X.shape[1]}, Target: {target_name}')
        print('#' * 70)

        # Use Jajpur feature names if available, else generic
        jajpur_names = ['pH', 'EC', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']
        if X.shape[1] == 12 and key == '1_jajpur':
            feat_names = jajpur_names
        else:
            feat_names = get_feature_names(X.shape[1])

        # KFold setup
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        cvss = list(kf.split(X))

        results = {}

        # === DE-FS ===
        print()
        print(f'--- DE Feature Selection ---')
        t0 = time.time()
        try:
            A1_de = run_DE_fs(X, y, cvss, feat_names)
            A1_de['Time'] = time.time() - t0
            results['DE-FS'] = A1_de
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  DE-FS FAILED: {e}')
            results['DE-FS'] = {'R2CV': np.nan, 'Error': str(e)[:100]}

        all_results[key] = results

        # === Comparison with Phase 1 baseline ===
        print()
        print(f'{"=" * 70}')
        print(f'  Comparison: Phase 1 (all features) vs Phase 2 (selected features)')
        print(f'{"=" * 70}')

        if key in baseline:
            print(f'  {"Method":<12} {"Phase1_R2CV":<14} {"Phase2_R2CV":<14} {"n_Feat":<8} {"Change":<10}')
            print(f'  {"-" * 60}')
            for method_baseline, r_baseline in baseline[key].items():
                r2cv_b1 = r_baseline.get('R2CV', np.nan)
                if method_baseline == 'DE' and 'DE-FS' in results:
                    r2cv_b2 = results['DE-FS']['R2CV']
                    nf = results['DE-FS']['N_Features']
                    change = r2cv_b2 - r2cv_b1
                    print(f'  {"DE-FS":<12} {r2cv_b1:<14.4f} {r2cv_b2:<14.4f} {nf:<8d} {change:>+.4f}')
                elif method_baseline == 'Best' and 'DE-FS' in results:
                    pass  # skip
        else:
            print(f'  (No Phase 1 baseline for {key})')

    # ===== Summary =====
    print()
    print('#' * 70)
    print('# PHASE 2 SUMMARY: Feature Selection Results')
    print('#' * 70)
    for key, results in all_results.items():
        print(f'\n{key}:')
        for method, r in results.items():
            if 'R2CV' in r and not np.isnan(r['R2CV']):
                print(f'  {method:<12} R2={r.get("R2",0):.4f} R2CV={r["R2CV"]:.4f} '
                      f'n_feat={r.get("N_Features","?")} time={r.get("Time",0):.1f}s')
                if 'KeptVars' in r:
                    print(f'          Kept: {r["KeptVars"]}')
                    if r.get('DroppedVars'):
                        print(f'          Dropped: {r["DroppedVars"]}')
            else:
                print(f'  {method:<12} FAILED')

    # Save
    save_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'feature_selection_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()