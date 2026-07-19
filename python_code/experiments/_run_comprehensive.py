"""
综合实验：3模型 × 6EA × 3集成方法 × 3数据集
=============================================
模型: MLP-lbfgs, 1D-CNN, XGBoost
EA:   DE, SHADE, CMA-ES, NRBO, BOA, HHO-Lite
集成: SimpleAvg, WeightedAvg, Stacking
数据集: Jajpur, Irish, AKH

输出:
  results/comprehensive_results.json     (所有指标汇总)
  results/{Model}_{Dataset}_*.csv        (收敛/散点/权重/误差/相关性)
"""
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import sys
import json
import time
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
os.makedirs(RESULTS_DIR, exist_ok=True)

from common_codes.models import MODEL_REGISTRY, get_model_config
from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt
from common_codes.ensemble.stacking import a4_ensemble_stacking
from sklearn.model_selection import KFold


# ==================== 工具函数 ====================

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


# ==================== 算法运行器 ====================

ALGO_NAME_MAP = {
    'DE': a4_DE_fitrnet_opt,
    'SHADE': a4_SHADE_fitrnet_opt,
    'CMA-ES': a4_CMAES_fitrnet_opt,
    'NRBO': a4_NRBO_fitrnet_opt,
    'BOA': a4_BOA_fitrnet_opt,
    'HHO-Lite': a4_HHO_Lite_fitrnet_opt,
}


def run_algorithm(name, X, y, max_evals, model_config):
    func = ALGO_NAME_MAP[name]
    print(f'  Running {name} (max_evals={max_evals})...', flush=True)
    t0 = time.time()
    Mdl, A1 = func(X, y, max_evals=max_evals, model_config=model_config)
    elapsed = time.time() - t0

    R2_alg = A1['R2']
    R2CV_alg = A1['R2CV']
    y_pred = Mdl.predict(X)
    _, RMSE_alg, MAE_alg = calc_metrics(y, y_pred)

    conv_key = next((k for k in A1 if 'convergence' in k.lower()), None)
    conv_data = A1.get(conv_key, []) if conv_key else []
    sampled = sample_convergence(conv_data, max_evals)

    params = A1.get('params', {})
    if not params:
        params = {k: A1[k] for k in model_config['param_names'] if k in A1}

    print(f'  {name} done: R2={R2_alg:.4f}  R2CV={R2CV_alg:.4f}  '
          f'RMSE={RMSE_alg:.3f}  MAE={MAE_alg:.3f}  Time={elapsed:.1f}s', flush=True)
    return {
        'R2': R2_alg, 'R2CV': R2CV_alg, 'RMSE': RMSE_alg, 'MAE': MAE_alg,
        'Time': elapsed, 'model': Mdl, 'params': params, 'convergence': sampled
    }


# ==================== 集成方法 ====================

def simple_avg(predictions):
    return np.mean(predictions, axis=0)


def weighted_avg(predictions, r2cv_scores):
    weights = np.array(r2cv_scores) / np.sum(r2cv_scores)
    return np.average(predictions, axis=0, weights=weights), weights


# ==================== CSV 输出 ====================

def save_convergence_csv(model_name, dataset_name, convergence_dict, max_gen):
    path = os.path.join(RESULTS_DIR, f'convergence_{model_name}_{dataset_name}.csv')
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation'] + algos)
        for gen in range(1, max_gen + 1):
            row = [gen]
            for a in algos:
                conv = convergence_dict.get(a, [])
                row.append(conv[min(gen - 1, len(conv) - 1)] if conv else 0.5)
            writer.writerow(row)
    print(f'  [CSV] Convergence: {os.path.basename(path)}')


def save_scatter_csv(model_name, dataset_name, y_actual, predictions_dict, ensemble_pred, ensemble_name):
    path = os.path.join(RESULTS_DIR, f'scatter_{model_name}_{dataset_name}.csv')
    ea_names = list(predictions_dict.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Actual'] + ea_names + [f'{ensemble_name}_Pred']
        writer.writerow(header)
        for i in range(len(y_actual)):
            row = [f'{y_actual[i]:.4f}']
            for a in ea_names:
                row.append(f'{predictions_dict[a][i]:.4f}')
            row.append(f'{ensemble_pred[i]:.4f}')
            writer.writerow(row)
    print(f'  [CSV] Scatter: {os.path.basename(path)}')


def save_weights_csv(model_name, dataset_name, weights, algo_names):
    path = os.path.join(RESULTS_DIR, f'weights_{model_name}_{dataset_name}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Weight'])
        for algo, w in zip(algo_names, weights):
            writer.writerow([algo, f'{w:.6f}'])
    print(f'  [CSV] Weights: {os.path.basename(path)}')


def save_errors_csv(model_name, dataset_name, y_actual, predictions_dict, ensemble_pred):
    path = os.path.join(RESULTS_DIR, f'errors_{model_name}_{dataset_name}.csv')
    ea_names = list(predictions_dict.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Sample'] + [f'{a}_AE' for a in ea_names] + ['Ensemble_AE']
        writer.writerow(header)
        for i in range(len(y_actual)):
            row = [i]
            for a in ea_names:
                row.append(f'{abs(y_actual[i] - predictions_dict[a][i]):.4f}')
            row.append(f'{abs(y_actual[i] - ensemble_pred[i]):.4f}')
            writer.writerow(row)
    print(f'  [CSV] Errors: {os.path.basename(path)}')


def save_correlation_csv(model_name, dataset_name, predictions_dict, algos):
    path = os.path.join(RESULTS_DIR, f'correlation_{model_name}_{dataset_name}.csv')
    preds = np.array([predictions_dict[a] for a in algos])
    corr = np.corrcoef(preds)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + algos)
        for i, a in enumerate(algos):
            row = [a] + [f'{corr[i, j]:.6f}' for j in range(len(algos))]
            writer.writerow(row)
    print(f'  [CSV] Correlation: {os.path.basename(path)}')


# ==================== 主函数 ====================

def main():
    print('=' * 70)
    print('综合实验: 3模型 × 6EA × 3集成方法 × 3数据集')
    print('=' * 70)

    # 模型列表（按顺序跑）
    model_names = ['MLP-lbfgs', '1D-CNN', 'XGBoost']

    datasets = {
        'Jajpur': '1_jajpur.npz',
        'Irish': '2_irish_river.npz',
        'AKH': '3_akh_wqi.npz'
    }

    # 每个算法/数据集的评估预算
    param_config = {
        'Jajpur': {'DE': 60, 'SHADE': 60, 'CMA-ES': 50, 'NRBO': 60, 'BOA': 60, 'HHO-Lite': 60},
        'Irish':  {'DE': 80, 'SHADE': 80, 'CMA-ES': 60, 'NRBO': 80, 'BOA': 80, 'HHO-Lite': 80},
        'AKH':    {'DE': 80, 'SHADE': 80, 'CMA-ES': 60, 'NRBO': 80, 'BOA': 80, 'HHO-Lite': 80}
    }

    ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    all_results = {}

    for model_name in model_names:
        model_config = get_model_config(model_name)
        print(f'\n{"#" * 70}')
        print(f'# 模型: {model_name}')
        print(f'# 参数: {model_config["param_names"]}')
        print(f'{"#" * 70}', flush=True)

        model_results = {}

        for dataset_name, file_name in datasets.items():
            print(f'\n{"=" * 60}')
            print(f'数据集: {dataset_name}  |  模型: {model_name}')
            print(f'{"=" * 60}', flush=True)

            data_path = os.path.join(DATASET_DIR, file_name)
            data = np.load(data_path, allow_pickle=True)
            X = data['X']
            y = data['y']
            print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features', flush=True)

            config = param_config[dataset_name]

            # ========== 1. 运行6个EA ==========
            results = {}
            for algo in ea_names:
                results[algo] = run_algorithm(algo, X, y, max_evals=config[algo], model_config=model_config)

            # ========== 收敛曲线CSV ==========
            convergence_dict = {}
            for a in ea_names:
                convergence_dict[a] = results[a]['convergence']
            max_gen = min(len(v) for v in convergence_dict.values() if v)
            max_gen = max(max_gen, 1)
            save_convergence_csv(model_name, dataset_name, convergence_dict, max_gen)

            # ========== 打印单算法结果 ==========
            print('\n' + '-' * 70)
            print(f"{'算法':<10} {'R2':>7} {'R2CV':>7} {'RMSE':>8} {'MAE':>8} {'Time(s)':>8}")
            print('-' * 70)
            for algo in ea_names:
                r = results[algo]
                print(f"{algo:<10} {r['R2']:>7.4f} {r['R2CV']:>7.4f} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['Time']:>8.1f}")

            # ========== 2. 集成实验 ==========
            print('\n' + '=' * 60)
            print('集成实验')
            print('=' * 60)

            r2cv_scores = [results[a]['R2CV'] for a in ea_names]
            models = [results[a]['model'] for a in ea_names]
            predictions = np.array([m.predict(X) for m in models])
            pred_dict = {a: results[a]['model'].predict(X) for a in ea_names}

            ensemble_results = {}

            # --- SimpleAvg ---
            y_simple = simple_avg(predictions)
            R2_s, RMSE_s, MAE_s = calc_metrics(y, y_simple)
            kf = KFold(n_splits=5, shuffle=True, random_state=1)
            r2cv_s_list = []
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                preds_test = np.array([m.predict(X_test) for m in models])
                y_pred_s_test = simple_avg(preds_test)
                R2CV_s_fold, _, _ = calc_metrics(y_test, y_pred_s_test)
                r2cv_s_list.append(R2CV_s_fold)
            R2CV_s = np.mean(r2cv_s_list)
            ensemble_results['SimpleAvg'] = {'R2': R2_s, 'R2CV': R2CV_s, 'RMSE': RMSE_s, 'MAE': MAE_s}
            print(f"  SimpleAvg:     R2={R2_s:.4f}  R2CV={R2CV_s:.4f}  RMSE={RMSE_s:.3f}  MAE={MAE_s:.3f}")

            # --- WeightedAvg ---
            y_weighted, weights = weighted_avg(predictions, r2cv_scores)
            R2_w, RMSE_w, MAE_w = calc_metrics(y, y_weighted)
            r2cv_w_list = []
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                preds_test = np.array([m.predict(X_test) for m in models])
                y_pred_w_test, _ = weighted_avg(preds_test, r2cv_scores)
                R2CV_w_fold, _, _ = calc_metrics(y_test, y_pred_w_test)
                r2cv_w_list.append(R2CV_w_fold)
            R2CV_w = np.mean(r2cv_w_list)
            ensemble_results['WeightedAvg'] = {'R2': R2_w, 'R2CV': R2CV_w, 'RMSE': RMSE_w, 'MAE': MAE_w}
            print(f"  WeightedAvg:   R2={R2_w:.4f}  R2CV={R2CV_w:.4f}  RMSE={RMSE_w:.3f}  MAE={MAE_w:.3f}")

            # --- Stacking ---
            print('  Running Stacking...', flush=True)
            try:
                Mdl_stack, A1_stack = a4_ensemble_stacking(X, y, model_config=model_config)
                y_stack = Mdl_stack.predict(X)
                R2_st, RMSE_st, MAE_st = calc_metrics(y, y_stack)
                R2CV_st = A1_stack['R2CV']
                ensemble_results['Stacking'] = {'R2': R2_st, 'R2CV': R2CV_st, 'RMSE': RMSE_st, 'MAE': MAE_st}
                print(f"  Stacking:      R2={R2_st:.4f}  R2CV={R2CV_st:.4f}  RMSE={RMSE_st:.3f}  MAE={MAE_st:.3f}")
            except Exception as e:
                print(f"  Stacking FAILED: {e}")
                ensemble_results['Stacking'] = {'R2': 0, 'R2CV': 0, 'RMSE': 0, 'MAE': 0, 'error': str(e)}

            # ========== 保存CSV ==========
            save_scatter_csv(model_name, dataset_name, y, pred_dict, y_weighted, 'WeightedAvg')
            save_correlation_csv(model_name, dataset_name, pred_dict, ea_names)
            save_weights_csv(model_name, dataset_name, weights, ea_names)
            save_errors_csv(model_name, dataset_name, y, pred_dict, y_weighted)

            # ========== 集成增益 ==========
            best_single_r2cv = max(r2cv_scores)
            best_single_algo = ea_names[r2cv_scores.index(best_single_r2cv)]
            best_ensemble_r2cv = max(r['R2CV'] for r in ensemble_results.values() if r['R2CV'] != 0)
            best_ensemble_method = max(
                (k for k in ensemble_results if ensemble_results[k]['R2CV'] != 0),
                key=lambda k: ensemble_results[k]['R2CV']
            )
            improvement_pp = round((best_ensemble_r2cv - best_single_r2cv) * 100, 2)

            # ========== 存储单算法结果 ==========
            single_save = {}
            for a in ea_names:
                r = results[a]
                single_save[a] = {'R2': r['R2'], 'R2CV': r['R2CV'],
                                  'RMSE': r['RMSE'], 'MAE': r['MAE'], 'Time': r['Time'],
                                  'params': {k: str(v) if not isinstance(v, (int, float, bool)) else v
                                             for k, v in r['params'].items()}}

            model_results[dataset_name] = {
                'single_results': single_save,
                'ensemble_results': ensemble_results,
                'best_single_r2cv': best_single_r2cv,
                'best_ensemble_r2cv': best_ensemble_r2cv,
                'best_single_algo': best_single_algo,
                'best_ensemble_method': best_ensemble_method,
                'improvement_pp': improvement_pp,
            }

        all_results[model_name] = model_results

    # ========== 汇总打印 ==========
    print('\n' + '=' * 70)
    print('所有实验汇总')
    print('=' * 70)
    print(f"{'模型':<12} {'数据集':<10} {'最佳单EA':<10} {'R2CV单':>8} "
          f"{'最佳集成':<14} {'R2CV集':>8} {'提升(pp)':>8}")
    print('-' * 75)
    for model_name in model_names:
        for dataset_name in datasets:
            r = all_results[model_name][dataset_name]
            print(f"{model_name:<12} {dataset_name:<10} {r['best_single_algo']:<10} "
                  f"{r['best_single_r2cv']:>8.4f} {r['best_ensemble_method']:<14} "
                  f"{r['best_ensemble_r2cv']:>8.4f} {r['improvement_pp']:>7.2f}pp")

    # ========== 保存JSON ==========
    json_path = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDone. JSON: {json_path}")
    print(f"Done. CSV files: {RESULTS_DIR}")


if __name__ == '__main__':
    main()