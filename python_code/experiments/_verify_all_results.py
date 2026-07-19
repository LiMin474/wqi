# -*- coding: utf-8 -*-
"""
Comprehensive verification of all dataset and result correctness.
Checks:
  1. Dataset integrity - samples, features, y range/mean/std per npz
  2. Results JSON - all metrics reasonable
  3. Convergence CSV - progressive improvement
  4. Scatter CSV - reasonable prediction correlation
  5. Weight CSV - weights sum to 1.0
  6. Correlation matrix CSV - symmetric, diagonal = 1.0
  7. Specific R2CV range checks per dataset
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
import csv
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
sys.path.insert(0, PROJECT_DIR)

DATASET_MAP = {
    'Jajpur': '1_jajpur.npz',
    'Irish': '2_irish_river.npz',
    'AKH': '3_akh_wqi.npz',
}

EA_NAMES = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
ALL_ALGO_NAMES = EA_NAMES + ['Bayesian']

EXPECTED = {
    'Jajpur': {'samples': 74, 'features': 12, 'r2cv_min': 0.9865, 'r2cv_max': 0.9921, 'ensemble_r2cv': 0.9998},
    'Irish':  {'samples': 501, 'features': 11, 'r2cv_min': 0.9546, 'r2cv_max': 0.9590, 'ensemble_r2cv': 0.9856},
    'AKH':    {'samples': 657, 'features': 10, 'r2cv_min': 0.7261, 'r2cv_max': 0.7523, 'ensemble_r2cv': 0.8424},
}

passed = 0
failed = 0
total_checks = 0


def load_results(model_name='MLP-lbfgs'):
    path = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
        print(f"[WARNING] comprehensive_results.json not found, falling back to {path}")
    with open(path) as f:
        data = json.load(f)
    if model_name in data:
        return data[model_name]
    return data


def check(name, condition, detail=''):
    global passed, failed, total_checks
    total_checks += 1
    status = 'PASS' if condition else 'FAIL'
    if condition:
        passed += 1
    else:
        failed += 1
    symbol = '[OK]' if condition else '[!!]'
    print(f'  {symbol} {status} | {name}' + (f' | {detail}' if detail else ''))


def print_separator(title):
    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print(f'{"=" * 70}')


print_separator('1. Dataset Integrity - dataset integrity check')

dataset_stats = {}
for ds_name, file_name in DATASET_MAP.items():
    print(f'\n--- {ds_name} ({file_name}) ---')
    filepath = os.path.join(DATASET_DIR, file_name)
    check(f'{ds_name}: npz file exists', os.path.isfile(filepath))

    try:
        d = np.load(filepath, allow_pickle=True)
        X, y = d['X'], d['y']
        n_samples, n_features = X.shape
        y_min, y_max = float(y.min()), float(y.max())
        y_mean = float(y.mean())
        y_std = float(y.std())
        n_nan_x = int(np.isnan(X).sum())
        n_nan_y = int(np.isnan(y).sum())
        n_inf_x = int(np.isinf(X).sum())
        n_inf_y = int(np.isinf(y).sum())

        dataset_stats[ds_name] = {
            'samples': n_samples, 'features': n_features,
            'y_min': y_min, 'y_max': y_max, 'y_mean': y_mean, 'y_std': y_std
        }

        print(f'    samples: {n_samples}, features: {n_features}')
        print(f'    y range: [{y_min:.4f}, {y_max:.4f}], mean: {y_mean:.4f}, std: {y_std:.4f}')
        print(f'    X NaN: {n_nan_x}, y NaN: {n_nan_y}, X Inf: {n_inf_x}, y Inf: {n_inf_y}')

        exp = EXPECTED[ds_name]
        check(f'{ds_name}: samples={exp["samples"]}', n_samples == exp['samples'], f'actual={n_samples}')
        check(f'{ds_name}: features={exp["features"]}', n_features == exp['features'], f'actual={n_features}')
        check(f'{ds_name}: X no NaN', n_nan_x == 0)
        check(f'{ds_name}: y no NaN', n_nan_y == 0)
        check(f'{ds_name}: X no Inf', n_inf_x == 0)
        check(f'{ds_name}: y no Inf', n_inf_y == 0)
        check(f'{ds_name}: y has valid range', y_min < y_max)
        check(f'{ds_name}: y std > 0', y_std > 0)
    except Exception as e:
        check(f'{ds_name}: load failed', False, str(e))

print_separator('2. Results JSON - metrics check')

model_name = sys.argv[1] if len(sys.argv) > 1 else 'MLP-lbfgs'
all_data = load_results(model_name)
json_path = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
check('JSON file exists', os.path.isfile(json_path) or os.path.isfile(os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')))

datasets_in_json = list(all_data.keys())
print(f'  Datasets in JSON: {datasets_in_json}')

for ds_name in DATASET_MAP:
    if ds_name not in all_data:
        check(f'{ds_name}: present in JSON', False, 'MISSING')
        continue

    d = all_data[ds_name]
    single = d.get('single_results', {})
    ensemble = d.get('ensemble_results', {})

    print(f'\n--- {ds_name} single algorithm metrics ---')
    print(f'  {"Algo":<10} {"R2":>8} {"R2CV":>8} {"RMSE":>10} {"MAE":>10} {"Time(s)":>8}')
    print(f'  {"-"*54}')
    for a in ALL_ALGO_NAMES:
        if a in single:
            r = single[a]
            print(f'  {a:<10} {r["R2"]:>8.4f} {r["R2CV"]:>8.4f} {r["RMSE"]:>10.3f} {r["MAE"]:>10.3f} {r["Time"]:>8.1f}')

    for algo in EA_NAMES:
        check(f'{ds_name}/{algo}: R2 exists', algo in single and 'R2' in single[algo])
        check(f'{ds_name}/{algo}: R2CV exists', algo in single and 'R2CV' in single[algo])
        if algo in single and 'R2CV' in single[algo]:
            r2cv = single[algo]['R2CV']
            check(f'{ds_name}/{algo}: 0<=R2CV<=1', 0 <= r2cv <= 1, f'R2CV={r2cv:.4f}')
        if algo in single and 'RMSE' in single[algo]:
            check(f'{ds_name}/{algo}: RMSE>0', single[algo]['RMSE'] > 0)
        if algo in single and 'MAE' in single[algo]:
            check(f'{ds_name}/{algo}: MAE>0', single[algo]['MAE'] > 0)

    wa = ensemble.get('WeightedAvg', {})
    if wa:
        check(f'{ds_name}/Ensemble: R2CV exists', 'R2CV' in wa)
        if 'R2CV' in wa:
            check(f'{ds_name}/Ensemble: 0<=R2CV<=1', 0 <= wa['R2CV'] <= 1, f'R2CV={wa["R2CV"]:.4f}')
        if 'R2' in wa:
            check(f'{ds_name}/Ensemble: R2 reasonable', 0 <= wa['R2'] <= 1)
        if 'RMSE' in wa:
            check(f'{ds_name}/Ensemble: RMSE>0', wa['RMSE'] > 0)
        if 'MAE' in wa:
            check(f'{ds_name}/Ensemble: MAE>0', wa['MAE'] > 0)

        best_single = max(single[a]['R2CV'] for a in EA_NAMES if a in single)
        check(f'{ds_name}: Ensemble R2CV > best single', wa['R2CV'] >= best_single,
              f'Ens={wa["R2CV"]:.4f} vs Best={best_single:.4f}')

    exp = EXPECTED[ds_name]
    r2cvs = [single[a]['R2CV'] for a in EA_NAMES if a in single]
    if r2cvs:
        actual_min, actual_max = min(r2cvs), max(r2cvs)
        check(f'{ds_name}: R2CV range [{exp["r2cv_min"]}~{exp["r2cv_max"]}]',
              exp['r2cv_min'] <= actual_min and actual_max <= exp['r2cv_max'],
              f'actual=[{actual_min:.4f}~{actual_max:.4f}]')
    if wa and 'R2CV' in wa:
        tol = 0.002
        check(f'{ds_name}: Ensemble R2CV ~ {exp["ensemble_r2cv"]}',
              abs(wa['R2CV'] - exp['ensemble_r2cv']) < tol,
              f'actual={wa["R2CV"]:.4f}, expected={exp["ensemble_r2cv"]:.4f}')

print_separator('3. Convergence CSV - convergence curve check')

for ds_name in DATASET_MAP:
    csv_path = os.path.join(RESULTS_DIR, f'convergence_{model_name}_{ds_name}.csv')
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(RESULTS_DIR, f'convergence_{ds_name}.csv')
    check(f'{ds_name}: convergence CSV exists', os.path.isfile(csv_path))
    if not os.path.isfile(csv_path):
        continue

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = header[1:]

        generations = []
        algo_series = {a: [] for a in cols}
        for row in reader:
            gen = int(row[0])
            generations.append(gen)
            for i, a in enumerate(cols):
                algo_series[a].append(float(row[i + 1]))

    n_gen = len(generations)
    print(f'  {ds_name}: {n_gen} generations, {len(cols)} algorithms')

    for i, a1 in enumerate(cols):
        for j, a2 in enumerate(cols):
            if i < j and algo_series[a1] == algo_series[a2]:
                n_identical = sum(1 for v1, v2 in zip(algo_series[a1], algo_series[a2]) if v1 == v2)
                check(f'{ds_name}: {a1}!={a2} (not homogeneous)', n_identical < n_gen,
                      f'{n_identical}/{n_gen} identical')

    for a in cols:
        series = algo_series[a]
        best_so_far = -float('inf')
        regressions = 0
        for val in series:
            if val < best_so_far - 1e-10:
                regressions += 1
            best_so_far = max(best_so_far, val)
        check(f'{ds_name}/{a}: convergence no regression', regressions == 0, f'regressions={regressions}')

        if len(series) >= 2:
            final_is_best = abs(max(series) - series[-1]) < 1e-10
            check(f'{ds_name}/{a}: final=best value', final_is_best,
                  f'final={series[-1]:.4f}, best={max(series):.4f}')

            improvement = series[-1] > series[0] + 1e-6
            check(f'{ds_name}/{a}: has improvement ({series[0]:.4f}->{series[-1]:.4f})',
                  improvement, f'first={series[0]:.4f} last={series[-1]:.4f}')

print_separator('4. Scatter CSV - prediction vs actual correlation check')

for ds_name in DATASET_MAP:
    csv_path = os.path.join(RESULTS_DIR, f'scatter_{model_name}_{ds_name}.csv')
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(RESULTS_DIR, f'scatter_{ds_name}.csv')
    check(f'{ds_name}: scatter CSV exists', os.path.isfile(csv_path))
    if not os.path.isfile(csv_path):
        continue

    df_data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_data.append(row)

    n_rows = len(df_data)
    actual = np.array([float(r['Actual']) for r in df_data])
    pred_col = 'WeightedAvg_Pred' if 'WeightedAvg_Pred' in df_data[0] else (ds_name + '_Pred') if (ds_name + '_Pred') in df_data[0] else 'Ensemble_Pred'
    ensemble_pred = np.array([float(r[pred_col]) for r in df_data])

    print(f'  {ds_name}: {n_rows} samples')

    r_ens = np.corrcoef(actual, ensemble_pred)[0, 1]
    check(f'{ds_name}: Ens vs Actual Pearson r>0.8', r_ens > 0.8, f'r={r_ens:.4f}')

    for a in EA_NAMES:
        if a not in df_data[0]:
            continue
        pred_a = np.array([float(r[a]) for r in df_data])
        r_a = np.corrcoef(actual, pred_a)[0, 1]
        check(f'{ds_name}/{a}: Pearson r>0.5', r_a > 0.5, f'r={r_a:.4f}')

    best_algo_r = max(
        np.corrcoef(actual, np.array([float(r[a]) for r in df_data]))[0, 1]
        for a in EA_NAMES if a in df_data[0]
    )
    check(f'{ds_name}: Ens r >= Best algo r * 0.95', r_ens >= best_algo_r * 0.95,
          f'Ens r={r_ens:.4f}, Best r={best_algo_r:.4f}')

    has_difficult = 'IsDifficult' in df_data[0]
    check(f'{ds_name}: IsDifficult column exists', has_difficult)
    if has_difficult:
        n_difficult = sum(1 for r in df_data if r['IsDifficult'] == 'True')
        check(f'{ds_name}: has difficult samples', n_difficult > 0,
              f'{n_difficult}/{n_rows} difficult samples')

    mae = np.mean(np.abs(actual - ensemble_pred))
    std_actual = np.std(actual)
    check(f'{ds_name}: Ens MAE reasonable (<0.5*std)', mae < 0.5 * std_actual,
          f'MAE={mae:.4f}, std(y)={std_actual:.4f}')

print_separator('5. Weight CSV - weight check')

for ds_name in DATASET_MAP:
    csv_path = os.path.join(RESULTS_DIR, f'weights_{model_name}_{ds_name}.csv')
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(RESULTS_DIR, f'weights_{ds_name}.csv')
    check(f'{ds_name}: weights CSV exists', os.path.isfile(csv_path))
    if not os.path.isfile(csv_path):
        continue

    weights = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(float(row['Weight']))

    weight_sum = sum(weights)
    check(f'{ds_name}: weights sum=1.0', abs(weight_sum - 1.0) < 1e-6, f'sum={weight_sum:.10f}')
    check(f'{ds_name}: 6 weights', len(weights) == 6, f'count={len(weights)}')
    check(f'{ds_name}: all weights>0', all(w > 0 for w in weights))
    check(f'{ds_name}: weight spread <=0.03', max(weights) - min(weights) <= 0.03,
          f'range=[{min(weights):.6f}~{max(weights):.6f}]')

    print(f'  {ds_name}: weights = {[f"{w:.6f}" for w in weights]}')

print_separator('6. Correlation Matrix CSV - correlation matrix check')

for ds_name in DATASET_MAP:
    csv_path = os.path.join(RESULTS_DIR, f'correlation_matrix_{ds_name}.csv')
    check(f'{ds_name}: correlation_matrix CSV exists', os.path.isfile(csv_path))
    if not os.path.isfile(csv_path):
        continue

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        algo_names = header[1:]

        matrix = []
        for row in reader:
            matrix.append([float(v) for v in row[1:]])

    mat = np.array(matrix)
    n = len(algo_names)
    check(f'{ds_name}: matrix is {n}x{n}', mat.shape == (n, n), f'shape={mat.shape}')
    check(f'{ds_name}: diagonal=1.0', all(abs(mat[i, i] - 1.0) < 1e-6 for i in range(n)))
    check(f'{ds_name}: symmetric matrix', np.allclose(mat, mat.T, atol=1e-6))
    check(f'{ds_name}: all coeffs in [-1,1]', np.all((mat >= -1) & (mat <= 1)))
    off_diag = mat[np.where(~np.eye(n, dtype=bool))]
    check(f'{ds_name}: off-diagonal != 1.0', np.all(off_diag < 0.999999),
          'some algos perfectly correlated')

print_separator('Verification Summary')

datasets_in_json = list(all_data.keys())
missing_datasets = [ds for ds in DATASET_MAP if ds not in datasets_in_json]
if missing_datasets:
    print(f'\n[WARN] Missing datasets in JSON: {", ".join(missing_datasets)}')
    print(f'   JSON only has: {datasets_in_json}')
    print(f'   Expected all three (Jajpur, Irish, AKH)')
    print(f'   Need to re-run _run_comprehensive.py for missing datasets')

print(f'\nTotal checks: {total_checks}')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
print(f'Pass rate: {passed/total_checks*100:.1f}%' if total_checks > 0 else 'N/A')

if failed > 0:
    print('\n[WARN] Some checks failed. Review [FAIL] items above.')
else:
    print('\n[PASS] All checks passed!')

print(f'\n{"=" * 70}')
