"""
仅运行Irish River数据集（6进化算法 + Bayesian + WeightedAvg集成）
保留全部中间数据供画图使用
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, sys, json, time, csv
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
os.makedirs(RESULTS_DIR, exist_ok=True)

from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt
from common_codes.benchmarks.Bayesian import a4_Bayesian_fitrnet_opt


def calc_metrics(y_true, y_pred):
    SST = np.sum((y_true - np.mean(y_true)) ** 2)
    SSE = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (SSE / SST) if SST != 0 else 0
    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
    MAE = np.mean(np.abs(y_true - y_pred))
    return R2, RMSE, MAE


def sample_convergence(sparse_conv, max_evals):
    if not sparse_conv:
        return []
    if not isinstance(sparse_conv[0], (list, tuple)):
        conv = list(sparse_conv)
        if len(conv) >= max_evals:
            return conv[:max_evals]
        return conv + [conv[-1]] * (max_evals - len(conv))
    dense = np.full(max_evals, np.nan)
    best = 0.0
    for ev, r2 in sparse_conv:
        best = max(best, r2)
        idx = min(int(ev) - 1, max_evals - 1)
        dense[idx] = best
    for i in range(1, max_evals):
        if np.isnan(dense[i]):
            dense[i] = dense[i - 1]
    if np.isnan(dense[0]):
        dense[0] = 0.0
    return dense.tolist()


ALGO_MAP = {
    'DE': a4_DE_fitrnet_opt, 'SHADE': a4_SHADE_fitrnet_opt,
    'CMA-ES': a4_CMAES_fitrnet_opt, 'NRBO': a4_NRBO_fitrnet_opt,
    'BOA': a4_BOA_fitrnet_opt, 'HHO-Lite': a4_HHO_Lite_fitrnet_opt,
    'Bayesian': a4_Bayesian_fitrnet_opt,
}


def main():
    dataset_name = 'Irish'
    file_name = '2_irish_river.npz'
    ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    # Irish有501个样本，适当增加评估预算
    config = {'DE': 30, 'SHADE': 40, 'CMA-ES': 30, 'NRBO': 40, 'BOA': 40, 'HHO-Lite': 40, 'Bayesian': 30}
    print(f'\n{"=" * 60}')
    print(f'Irish-River-CCME: 6 EA + Bayesian + WeightedAvg集成')
    print(f'{"=" * 60}')

    data_path = os.path.join(DATASET_DIR, file_name)
    data = np.load(data_path, allow_pickle=True)
    X, y = data['X'], data['y']
    print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features\n')

    # ========== 运行7个算法 ==========
    results = {}
    all_algo_names = list(ALGO_MAP.keys())
    for algo in all_algo_names:
        func = ALGO_MAP[algo]
        max_ev = config[algo]
        print(f'  Running {algo} (max_evals={max_ev})...', flush=True)
        t0 = time.time()
        Mdl, A1 = func(X, y, max_evals=max_ev)
        elapsed = time.time() - t0

        R2_alg = A1['R2']
        R2CV_alg = A1['R2CV']
        y_pred = Mdl.predict(X)
        _, RMSE_alg, MAE_alg = calc_metrics(y, y_pred)

        conv_key = next((k for k in A1 if 'convergence' in k.lower()), None)
        conv_data = A1.get(conv_key, []) if conv_key else []
        sampled = sample_convergence(conv_data, max_ev)

        params = {
            'NumLayers': A1.get('NumLayers', 0),
            'Layer_1': A1.get('Layer_1', 0),
            'Layer_2': A1.get('Layer_2', 0),
            'Activation': A1.get('Activation', ''),
            'Alpha': A1.get('Alpha', 0.0),
        }

        print(f'  {algo} done: R2={R2_alg:.4f}  R2CV={R2CV_alg:.4f}  RMSE={RMSE_alg:.3f}  MAE={MAE_alg:.3f}  Time={elapsed:.1f}s\n')
        results[algo] = {
            'R2': R2_alg, 'R2CV': R2CV_alg, 'RMSE': RMSE_alg, 'MAE': MAE_alg,
            'Time': elapsed, 'model': Mdl, 'params': params, 'convergence': sampled
        }

    # ========== 打印结果表 ==========
    print('-' * 70)
    print(f"{'算法':<10} {'R2':>7} {'R2CV':>7} {'RMSE':>8} {'MAE':>8} {'Time(s)':>8}")
    print('-' * 70)
    for algo in all_algo_names:
        r = results[algo]
        print(f"{algo:<10} {r['R2']:>7.4f} {r['R2CV']:>7.4f} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['Time']:>8.1f}")

    # ========== 收敛曲线CSV ==========
    conv_path = os.path.join(RESULTS_DIR, f'convergence_{dataset_name}.csv')
    convergence_dict = {}
    for a in all_algo_names:
        convergence_dict[a] = results[a]['convergence']
    max_gen = min(len(v) for v in convergence_dict.values() if v)
    max_gen = max(max_gen, 1)
    with open(conv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation'] + all_algo_names)
        for gen in range(1, max_gen + 1):
            row = [gen]
            for a in all_algo_names:
                conv = convergence_dict.get(a, [])
                row.append(conv[min(gen - 1, len(conv) - 1)] if conv else 0.5)
            writer.writerow(row)
    print(f'\n[CSV] Convergence saved: {conv_path}')

    # ========== 集成（仅6个EA） ==========
    print('\n' + '=' * 60)
    print('WeightedAvg 集成（6进化算法）')
    print('=' * 60)

    r2cv_scores = [results[a]['R2CV'] for a in ea_names]
    models = [results[a]['model'] for a in ea_names]
    predictions = np.array([m.predict(X) for m in models])
    pred_dict = {a: results[a]['model'].predict(X) for a in ea_names}

    # WeightedAvg
    weights = np.array(r2cv_scores) / np.sum(r2cv_scores)
    y_weighted = np.average(predictions, axis=0, weights=weights)
    R2_w, RMSE_w, MAE_w = calc_metrics(y, y_weighted)

    # R²CV for WeightedAvg (5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    r2cv_w_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        preds_test = np.array([m.predict(X_test) for m in models])
        w_test = np.array(r2cv_scores) / np.sum(r2cv_scores)
        y_pred_w_test = np.average(preds_test, axis=0, weights=w_test)
        R2CV_w_fold, _, _ = calc_metrics(y_test, y_pred_w_test)
        r2cv_w_list.append(R2CV_w_fold)
    R2CV_w = np.mean(r2cv_w_list)
    print(f"  WeightedAvg: R2={R2_w:.4f}  R2CV={R2CV_w:.4f}  RMSE={RMSE_w:.3f}  MAE={MAE_w:.3f}")

    best_single_r2cv = max(r2cv_scores)
    best_single_algo = ea_names[r2cv_scores.index(best_single_r2cv)]
    improvement_pp = round((R2CV_w - best_single_r2cv) * 100, 2)
    improvement_pct = round((R2CV_w - best_single_r2cv) / best_single_r2cv * 100, 2)
    print(f"  最佳单算法: {best_single_algo} (R2CV={best_single_r2cv:.4f})")
    print(f"  集成提升: {improvement_pp}pp ({improvement_pct}%)")

    # ========== 保存CSV ==========
    # 1. 散点数据
    scatter_path = os.path.join(RESULTS_DIR, f'scatter_{dataset_name}.csv')
    errors_dict = {a: np.abs(y - pred) for a, pred in pred_dict.items()}
    threshold = np.mean(list(errors_dict.values())) + 2 * np.std(list(errors_dict.values()))
    with open(scatter_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Actual'] + ea_names + ['WeightedAvg_Pred', 'IsDifficult'])
        for i in range(len(y)):
            row = [f'{y[i]:.4f}']
            for a in ea_names:
                row.append(f'{pred_dict[a][i]:.4f}')
            row.append(f'{y_weighted[i]:.4f}')
            avg_error = np.mean([errors_dict[a][i] for a in ea_names])
            row.append('True' if avg_error > threshold else 'False')
            writer.writerow(row)
    print(f'[CSV] Scatter saved: {scatter_path}')

    # 2. 权重数据
    weights_path = os.path.join(RESULTS_DIR, f'weights_{dataset_name}.csv')
    with open(weights_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Weight'])
        for i, a in enumerate(ea_names):
            writer.writerow([a, f'{weights[i]:.6f}'])
    print(f'[CSV] Weights saved: {weights_path}')

    # 3. 误差数据
    errors_path = os.path.join(RESULTS_DIR, f'errors_{dataset_name}.csv')
    with open(errors_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample'] + [f'{a}_AE' for a in ea_names] + ['Ensemble_AE'])
        for i in range(len(y)):
            row = [i]
            for a in ea_names:
                row.append(f'{abs(y[i] - pred_dict[a][i]):.4f}')
            row.append(f'{abs(y[i] - y_weighted[i]):.4f}')
            writer.writerow(row)
    print(f'[CSV] Errors saved: {errors_path}')

    # 4. 相关性矩阵
    corr_path = os.path.join(RESULTS_DIR, f'correlation_matrix_{dataset_name}.csv')
    preds = np.array([pred_dict[a] for a in ea_names])
    corr = np.corrcoef(preds)
    with open(corr_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + ea_names)
        for i, a in enumerate(ea_names):
            row = [a] + [f'{corr[i, j]:.6f}' for j in range(len(ea_names))]
            writer.writerow(row)
    print(f'[CSV] Correlation matrix saved: {corr_path}')

    # ========== 保存JSON ==========
    single_save = {}
    for a in all_algo_names:
        r = results[a]
        single_save[a] = {'R2': r['R2'], 'R2CV': r['R2CV'],
                          'RMSE': r['RMSE'], 'MAE': r['MAE'], 'Time': r['Time']}

    raw_convergence = {}
    for a in all_algo_names:
        raw_convergence[a] = results[a]['convergence']

    all_results = {
        dataset_name: {
            'single_results': single_save,
            'ensemble_results': {
                'WeightedAvg': {'R2': R2_w, 'R2CV': R2CV_w, 'RMSE': RMSE_w, 'MAE': MAE_w}
            },
            'best_single_r2cv': best_single_r2cv,
            'best_ensemble_r2cv': R2CV_w,
            'best_single_algo': best_single_algo,
            'best_ensemble_method': 'WeightedAvg',
            'improvement_pp': improvement_pp,
            'improvement_pct': improvement_pct,
            'convergence': raw_convergence,
        }
    }

    json_path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
    merged = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
    merged[dataset_name] = all_results[dataset_name]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f'\n[JSON] Results saved: {json_path}')
    print(f'[DONE] Irish 实验完成！所有CSV/JSON文件保存至: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
