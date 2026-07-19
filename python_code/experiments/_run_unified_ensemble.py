"""
统一集成实验：6进化算法 + Bayesian对比 在3个数据集上
========================================================
输出：
  - unified_ensemble_results.json  (所有指标)
  - convergence_{Dataset}.csv      (收敛曲线每代R²CV)
  - scatter_{Dataset}.csv          (各样本预测值 + 集成预测)
  - weights_{Dataset}.csv          (WeightedAvg 集成权重)
  - errors_{Dataset}.csv           (每样本绝对误差，用于箱线图)
  - correlation_matrix_{Dataset}.csv (算法预测相关性)
  - pareto_chart.csv               (效率图数据)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import sys
import json
import time
import csv
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 导入真实算法实现 ====================
from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt
from common_codes.benchmarks.Bayesian import a4_Bayesian_fitrnet_opt


# ==================== 工具函数 ====================

def calc_metrics(y_true, y_pred):
    SST = np.sum((y_true - np.mean(y_true)) ** 2)
    SSE = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (SSE / SST) if SST != 0 else 0
    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
    MAE = np.mean(np.abs(y_true - y_pred))
    return R2, RMSE, MAE


def sample_convergence(sparse_conv, max_evals):
    """将稀疏收敛序列（eval_num, R²CV元组列表）转换为密集序列"""
    if not sparse_conv:
        return []
    # Bayesian: 已经是密集序列（平铺R²CV值列表）
    if not isinstance(sparse_conv[0], (list, tuple)):
        conv = list(sparse_conv)
        if len(conv) >= max_evals:
            return conv[:max_evals]
        return conv + [conv[-1]] * (max_evals - len(conv))
    # EA: 稀疏元组列表 → 密集填充
    dense = np.full(max_evals, np.nan)
    best = 0.0
    for ev, r2 in sparse_conv:
        best = max(best, r2)
        idx = min(int(ev) - 1, max_evals - 1)
        dense[idx] = best
    # 前向填充NaN
    for i in range(1, max_evals):
        if np.isnan(dense[i]):
            dense[i] = dense[i - 1]
    if np.isnan(dense[0]):
        dense[0] = 0.0
    return dense.tolist()


# ==================== 真实算法运行器（分派器） ====================

ALGO_NAME_MAP = {
    'DE': a4_DE_fitrnet_opt,
    'SHADE': a4_SHADE_fitrnet_opt,
    'CMA-ES': a4_CMAES_fitrnet_opt,
    'NRBO': a4_NRBO_fitrnet_opt,
    'BOA': a4_BOA_fitrnet_opt,
    'HHO-Lite': a4_HHO_Lite_fitrnet_opt,
    'Bayesian': a4_Bayesian_fitrnet_opt,
}


def run_real_algorithm(name, X, y, max_evals):
    """调用真实算法实现，返回统一格式的结果字典"""
    func = ALGO_NAME_MAP[name]
    print(f'  Running {name} (max_evals={max_evals})...', flush=True)
    t0 = time.time()
    Mdl, A1 = func(X, y, max_evals=max_evals)
    elapsed = time.time() - t0

    R2_alg = A1['R2']
    R2CV_alg = A1['R2CV']
    y_pred = Mdl.predict(X)
    _, RMSE_alg, MAE_alg = calc_metrics(y, y_pred)

    # 智能检测收敛键名（不同算法关键词不同）
    conv_key = next((k for k in A1 if 'convergence' in k.lower()), None)
    conv_data = A1.get(conv_key, []) if conv_key else []
    sampled = sample_convergence(conv_data, max_evals)

    params = {
        'NumLayers': A1.get('NumLayers', 0),
        'Layer_1': A1.get('Layer_1', 0),
        'Layer_2': A1.get('Layer_2', 0),
        'Activation': A1.get('Activation', ''),
        'Alpha': A1.get('Alpha', 0.0),
    }

    print(f'  {name} done: R²={R2_alg:.4f}  R²CV={R2CV_alg:.4f}  RMSE={RMSE_alg:.3f}  MAE={MAE_alg:.3f}  Time={elapsed:.1f}s', flush=True)
    return {
        'R2': R2_alg, 'R2CV': R2CV_alg, 'RMSE': RMSE_alg, 'MAE': MAE_alg,
        'Time': elapsed, 'model': Mdl, 'params': params, 'convergence': sampled
    }


# ==================== 集成方法 ====================

def weighted_avg(predictions, r2cv_scores):
    weights = np.array(r2cv_scores) / np.sum(r2cv_scores)
    return np.average(predictions, axis=0, weights=weights), weights


# ==================== 输出CSV辅助函数 ====================

def save_convergence_csv(dataset_name, convergence_dict, max_gen):
    path = os.path.join(RESULTS_DIR, f'convergence_{dataset_name}.csv')
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite', 'Bayesian']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation'] + algos)
        for gen in range(1, max_gen + 1):
            row = [gen]
            for a in algos:
                conv = convergence_dict.get(a, [])
                row.append(conv[min(gen - 1, len(conv) - 1)] if conv else 0.5)
            writer.writerow(row)
    print(f'  [CSV] Convergence saved: {path}')


def save_scatter_csv(dataset_name, y_actual, predictions_dict, ensemble_pred, ensemble_name):
    path = os.path.join(RESULTS_DIR, f'scatter_{dataset_name}.csv')
    ea_names = list(predictions_dict.keys())
    errors = {a: np.abs(y_actual - pred) for a, pred in predictions_dict.items()}
    threshold = np.mean(list(errors.values())) + 2 * np.std(list(errors.values()))
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Actual'] + ea_names + [f'{ensemble_name}_Pred', 'IsDifficult']
        writer.writerow(header)
        for i in range(len(y_actual)):
            row = [f'{y_actual[i]:.4f}']
            for a in ea_names:
                row.append(f'{predictions_dict[a][i]:.4f}')
            row.append(f'{ensemble_pred[i]:.4f}')
            avg_error = np.mean([errors[a][i] for a in ea_names])
            row.append('True' if avg_error > threshold else 'False')
            writer.writerow(row)
    print(f'  [CSV] Scatter saved: {path}')


def save_weights_csv(dataset_name, weights):
    path = os.path.join(RESULTS_DIR, f'weights_{dataset_name}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Weight'])
        for algo, w in weights:
            writer.writerow([algo, f'{w:.6f}'])
    print(f'  [CSV] Weights saved: {path}')


def save_errors_csv(dataset_name, y_actual, predictions_dict, ensemble_pred):
    """保存每样本绝对误差，用于MAE箱线图"""
    path = os.path.join(RESULTS_DIR, f'errors_{dataset_name}.csv')
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
    print(f'  [CSV] Errors saved: {path}')


def save_correlation_csv(dataset_name, predictions_dict, algos):
    path = os.path.join(RESULTS_DIR, f'correlation_matrix_{dataset_name}.csv')
    preds = np.array([predictions_dict[a] for a in algos])
    corr = np.corrcoef(preds)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + algos)
        for i, a in enumerate(algos):
            row = [a] + [f'{corr[i, j]:.6f}' for j in range(len(algos))]
            writer.writerow(row)
    print(f'  [CSV] Correlation matrix saved: {path}')


def save_pareto_csv(all_results, datasets):
    path = os.path.join(RESULTS_DIR, 'pareto_chart.csv')
    algos = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite', 'Bayesian']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Algorithm', 'R2CV', 'Time(s)'])
        for ds in datasets:
            r = all_results[ds]['single_results']
            for a in algos:
                writer.writerow([ds, a, f'{r[a]["R2CV"]:.6f}', f'{r[a]["Time"]:.2f}'])
    print(f'  [CSV] Pareto chart saved: {path}')


# ==================== 主函数 ====================

def main():
    print('=' * 60)
    print('统一集成实验：6进化算法 + Bayesian × 3数据集')
    print('记录: R², R²CV, RMSE, MAE, Time + 收敛/散点/权重/误差/相关性CSV')
    print('=' * 60)

    datasets = {
        'Jajpur': '1_jajpur.npz',
        'Irish': '2_irish_river.npz',
        'AKH': '3_akh_wqi.npz'
    }

    # 每个算法/数据集的评估预算（max_evals）
    # 默认60次评估，复杂数据集适当增加
    param_config = {
        'Jajpur': {'DE': 50, 'SHADE': 50, 'CMA-ES': 50, 'NRBO': 50, 'BOA': 50, 'HHO-Lite': 50, 'Bayesian': 50},
        'Irish':  {'DE': 60, 'SHADE': 60, 'CMA-ES': 60, 'NRBO': 60, 'BOA': 60, 'HHO-Lite': 60, 'Bayesian': 60},
        'AKH':    {'DE': 60, 'SHADE': 60, 'CMA-ES': 60, 'NRBO': 60, 'BOA': 60, 'HHO-Lite': 60, 'Bayesian': 60}
    }

    all_results = {}
    all_convergence = {}
    ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']

    for dataset_name, file_name in datasets.items():
        print(f'\n{"=" * 60}')
        print(f'数据集: {dataset_name}')
        print(f'{"=" * 60}', flush=True)

        data_path = os.path.join(DATASET_DIR, file_name)
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features', flush=True)

        config = param_config[dataset_name]

        # ========== 运行7个算法（6 EA + Bayesian） ==========
        results = {}
        all_algo_names = list(ALGO_NAME_MAP.keys())
        for algo in all_algo_names:
            results[algo] = run_real_algorithm(algo, X, y, max_evals=config[algo])

        # ========== 收敛曲线CSV ==========
        convergence_dict = {}
        for a in all_algo_names:
            convergence_dict[a] = results[a]['convergence']
        max_gen = min(len(v) for v in convergence_dict.values() if v)
        max_gen = max(max_gen, 1)
        all_convergence[dataset_name] = {a: conv[:max_gen] for a, conv in convergence_dict.items()}
        save_convergence_csv(dataset_name, all_convergence[dataset_name], max_gen)

        # ========== 打印单算法结果 ==========
        print('\n' + '-' * 60)
        print(f"{'算法':<10} {'R²':>7} {'R²CV':>7} {'RMSE':>8} {'MAE':>8} {'Time(s)':>8}")
        print('-' * 60)
        for algo in all_algo_names:
            r = results[algo]
            print(f"{algo:<10} {r['R2']:>7.4f} {r['R2CV']:>7.4f} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['Time']:>8.1f}")

        # ========== 集成（仅6个EA，不含Bayesian） ==========
        print('\n' + '=' * 60)
        print('集成实验（6进化算法，不含Bayesian）')
        print('=' * 60)

        r2cv_scores = [results[a]['R2CV'] for a in ea_names]
        models = [results[a]['model'] for a in ea_names]
        # 全量预测
        predictions = np.array([m.predict(X) for m in models])
        pred_dict = {a: results[a]['model'].predict(X) for a in ea_names}

        # --- WeightedAvg ---
        y_weighted, weights = weighted_avg(predictions, r2cv_scores)
        R2_w, RMSE_w, MAE_w = calc_metrics(y, y_weighted)
        # R²CV for WeightedAvg (5-fold)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        r2cv_w_list = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            preds_test = np.array([m.predict(X_test) for m in models])
            y_pred_w_test, _ = weighted_avg(preds_test, r2cv_scores)
            R2CV_w_fold, _, _ = calc_metrics(y_test, y_pred_w_test)
            r2cv_w_list.append(R2CV_w_fold)
        R2CV_w = np.mean(r2cv_w_list)
        print(f"  WeightedAvg: R²={R2_w:.4f}  R²CV={R2CV_w:.4f}  RMSE={RMSE_w:.3f}  MAE={MAE_w:.3f}")

        # ========== 集成增益 ==========
        best_single_r2cv = max(r2cv_scores)
        best_single_algo = ea_names[r2cv_scores.index(best_single_r2cv)]
        improvement_pp = round((R2CV_w - best_single_r2cv) * 100, 2)
        improvement_pct = round((R2CV_w - best_single_r2cv) / best_single_r2cv * 100, 2)

        # ========== 保存CSV ==========
        save_scatter_csv(dataset_name, y, pred_dict, y_weighted, 'WeightedAvg')
        save_correlation_csv(dataset_name, pred_dict, ea_names)
        save_weights_csv(dataset_name, list(zip(ea_names, weights)))
        save_errors_csv(dataset_name, y, pred_dict, y_weighted)

        # ========== 存储结果 ==========
        single_save = {}
        for a in all_algo_names:
            r = results[a]
            single_save[a] = {'R2': r['R2'], 'R2CV': r['R2CV'],
                              'RMSE': r['RMSE'], 'MAE': r['MAE'], 'Time': r['Time']}

        all_results[dataset_name] = {
            'single_results': single_save,
            'ensemble_results': {
                'WeightedAvg': {'R2': R2_w, 'R2CV': R2CV_w, 'RMSE': RMSE_w, 'MAE': MAE_w}
            },
            'best_single_r2cv': best_single_r2cv,
            'best_ensemble_r2cv': R2CV_w,
            'best_single_algo': best_single_algo,
            'best_ensemble_method': 'WeightedAvg',
            'improvement_pp': improvement_pp,
            'improvement_pct': improvement_pct
        }

    # ========== 帕累托CSV ==========
    save_pareto_csv(all_results, list(datasets.keys()))

    # ========== 汇总打印 ==========
    print('\n' + '=' * 60)
    print('所有数据集汇总')
    print('=' * 60)
    header = f"{'数据集':<12} {'最佳单算法':<10} {'R²CV单':>8} {'集成方法':<14} {'R²CV集':>8} {'提升(pp)':>8} {'提升(%)':>8}"
    print(header)
    print('-' * 70)
    for dn in datasets:
        r = all_results[dn]
        print(f"{dn:<12} {r['best_single_algo']:<10} {r['best_single_r2cv']:>8.4f} "
              f"{r['best_ensemble_method']:<14} {r['best_ensemble_r2cv']:>8.4f} "
              f"{r['improvement_pp']:>7.2f}pp {r['improvement_pct']:>7.2f}%")

    # ========== 保存JSON ==========
    json_path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON结果保存至: {json_path}")
    print(f"✅ 所有CSV文件保存至: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
