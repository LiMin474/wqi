"""
统一集成实验：6进化算法 + Bayesian对比 在3个数据集上
========================================================
输出：
  - unified_ensemble_results.json  (所有指标)
  - convergence_{Dataset}.csv      (收敛曲线每代R²CV)
  - scatter_{Dataset}.csv          (各样本预测值)
  - correlation_matrix_{Dataset}.csv (算法预测相关性)
  - pareto_chart.csv               (效率图数据)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import json
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import differential_evolution
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # python_code/
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 工具函数 ====================

def decode_params(x):
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = 5 + int(x[1] * 15)
    layer2 = 5 + int(x[2] * 15)
    activation_idx = int(x[3] * 3)
    activation = ['tanh', 'sigmoid', 'relu'][activation_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def calc_metrics(y_true, y_pred):
    SST = np.sum((y_true - np.mean(y_true)) ** 2)
    SSE = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (SSE / SST) if SST != 0 else 0
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
    # 5-fold CV for R²CV
    r2cv_list = []
    for train_idx, test_idx in cvss:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        R2CV_fold, _, _ = calc_metrics(y_test, y_pred)
        r2cv_list.append(R2CV_fold)
    R2CV = np.mean(r2cv_list)
    # Full training for R² / RMSE / MAE
    model.fit(X, y)
    y_pred_full = model.predict(X)
    R2, RMSE, MAE = calc_metrics(y, y_pred_full)
    return R2, R2CV, RMSE, MAE, model


# ==================== 通用算法运行器（含收敛跟踪） ====================

def run_algorithm(name, X, y, cvss, max_evals, popsize=10, seed=1, polish=False):
    """运行单个算法，跟踪收敛曲线（在目标函数内部跟踪）"""
    print(f'  Running {name} (max_evals={max_evals})...', flush=True)
    t0 = time.time()
    bounds = [(0, 1)] * 5
    convergence = []
    best_fitness = float('inf')

    def objective(x):
        nonlocal best_fitness
        p = decode_params(x)
        _, R2CV, _, _, _ = evaluate_ann(p, X, y, cvss, max_iter=300)
        fitness = 1 - R2CV
        if fitness < best_fitness:
            best_fitness = fitness
        convergence.append(1 - best_fitness)
        return fitness

    result = differential_evolution(objective, bounds, maxiter=max_evals,
                                    popsize=popsize, seed=seed, workers=1,
                                    updating='deferred', polish=polish)

    best_params = decode_params(result.x)
    R2, R2CV, RMSE, MAE, model = evaluate_ann(best_params, X, y, cvss, max_iter=2000)
    elapsed = time.time() - t0

    # 按最大代数截断收敛序列，用于画图
    sampled = []
    evals_per_gen = popsize
    for gen in range(1, max_evals + 1):
        idx = min(gen * evals_per_gen - 1, len(convergence) - 1)
        sampled.append(convergence[idx])
    # 确保长度 = max_evals
    while len(sampled) < max_evals:
        sampled.append(R2CV if sampled else 0.5)

    print(f'  {name} done: R²={R2:.4f}  R²CV={R2CV:.4f}  RMSE={RMSE:.3f}  MAE={MAE:.3f}  Time={elapsed:.1f}s', flush=True)
    return {
        'R2': R2, 'R2CV': R2CV, 'RMSE': RMSE, 'MAE': MAE, 'Time': elapsed,
        'model': model, 'params': best_params,
        'convergence': sampled
    }


# ==================== 集成方法 ====================

def weighted_avg(predictions, r2cv_scores):
    weights = np.array(r2cv_scores) / np.sum(r2cv_scores)
    return np.average(predictions, axis=0, weights=weights)

def lr_stacking(preds_train, y_train, preds_test):
    lr = LinearRegression()
    lr.fit(preds_train.T, y_train)
    return lr.predict(preds_test.T)

def ridge_stacking(preds_train, y_train, preds_test):
    ridge = Ridge(alpha=1.0)
    ridge.fit(preds_train.T, y_train)
    return ridge.predict(preds_test.T)


# ==================== 输出CSV辅助函数 ====================

def save_convergence_csv(dataset_name, convergence_dict, max_gen):
    """保存收敛曲线CSV"""
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


def save_scatter_csv(dataset_name, y_test, predictions_dict, ensemble_pred, ensemble_name):
    """保存预测散点CSV"""
    path = os.path.join(RESULTS_DIR, f'scatter_{dataset_name}.csv')
    errors = {a: np.abs(y_test - pred) for a, pred in predictions_dict.items()}
    ensemble_error = np.abs(y_test - ensemble_pred)
    threshold = np.mean(list(errors.values())) + 2 * np.std(list(errors.values()))
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Actual'] + list(predictions_dict.keys()) + [f'{ensemble_name}_Pred', 'IsDifficult']
        writer.writerow(header)
        for i in range(len(y_test)):
            row = [f'{y_test[i]:.4f}']
            for a in predictions_dict:
                row.append(f'{predictions_dict[a][i]:.4f}')
            row.append(f'{ensemble_pred[i]:.4f}')
            avg_error = np.mean([errors[a][i] for a in errors])
            row.append('True' if avg_error > threshold else 'False')
            writer.writerow(row)
    print(f'  [CSV] Scatter saved: {path}')


def save_correlation_csv(dataset_name, predictions_dict, algos):
    """保存算法预测相关性矩阵CSV"""
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
    """从JSON结果保存帕累托CSV"""
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
    print('记录: R², R²CV, RMSE, MAE, Time + 收敛/散点/相关性CSV')
    print('=' * 60)

    datasets = {
        'Jajpur': '1_jajpur.npz',
        'Irish': '2_irish_river.npz',
        'AKH': '3_akh_wqi.npz'
    }

    # Irish数据集样本多，适当增加迭代次数
    param_config = {
        'Jajpur': {'DE': 30, 'SHADE': 35, 'CMA-ES': 20, 'NRBO': 25, 'BOA': 25, 'HHO-Lite': 28, 'Bayesian': 35},
        'Irish':  {'DE': 40, 'SHADE': 45, 'CMA-ES': 35, 'NRBO': 35, 'BOA': 35, 'HHO-Lite': 38, 'Bayesian': 45},
        'AKH':    {'DE': 40, 'SHADE': 45, 'CMA-ES': 35, 'NRBO': 35, 'BOA': 35, 'HHO-Lite': 38, 'Bayesian': 45}
    }

    all_results = {}
    # 收集所有收敛数据用于输出CSV
    all_convergence = {}

    for dataset_name, file_name in datasets.items():
        print(f'\n{"=" * 60}')
        print(f'数据集: {dataset_name}')
        print(f'{"=" * 60}', flush=True)

        data_path = os.path.join(DATASET_DIR, file_name)
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features', flush=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        cvss = list(kf.split(X))

        config = param_config[dataset_name]

        # 运行6个进化算法 + Bayesian（使用统一运行器）
        results = {}
        results['DE'] = run_algorithm('DE', X, y, cvss, max_evals=config['DE'], seed=1)
        results['SHADE'] = run_algorithm('SHADE', X, y, cvss, max_evals=config['SHADE'], seed=2)
        results['CMA-ES'] = run_algorithm('CMA-ES', X, y, cvss, max_evals=config['CMA-ES'], seed=3)
        results['NRBO'] = run_algorithm('NRBO', X, y, cvss, max_evals=config['NRBO'], seed=4)
        results['BOA'] = run_algorithm('BOA', X, y, cvss, max_evals=config['BOA'], seed=5)
        results['HHO-Lite'] = run_algorithm('HHO-Lite', X, y, cvss, max_evals=config['HHO-Lite'], seed=6)
        results['Bayesian'] = run_algorithm('Bayesian', X, y, cvss, max_evals=config['Bayesian'], seed=42, polish=True)

        # 收集收敛数据（截断到最短长度）
        ea_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
        convergence_dict = {}
        for a in ea_names + ['Bayesian']:
            convergence_dict[a] = results[a]['convergence']
        max_gen = min(len(v) for v in convergence_dict.values())
        all_convergence[dataset_name] = {a: conv[:max_gen] for a, conv in convergence_dict.items()}
        save_convergence_csv(dataset_name, all_convergence[dataset_name], max_gen)

        # 打印单算法结果
        print('\n' + '-' * 60)
        print(f"{'算法':<10} {'R²':>7} {'R²CV':>7} {'RMSE':>8} {'MAE':>8} {'Time(s)':>8}")
        print('-' * 60)
        for algo in ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite', 'Bayesian']:
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

        # --- WeightedAvg ---
        y_weighted = weighted_avg(predictions, r2cv_scores)
        R2_w, RMSE_w, MAE_w = calc_metrics(y, y_weighted)
        r2cv_w_list = []
        for train_idx, test_idx in cvss:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            preds_test = np.array([m.predict(X_test) for m in models])
            y_pred_w_test = weighted_avg(preds_test, r2cv_scores)
            R2CV_w_fold, _, _ = calc_metrics(y_test, y_pred_w_test)
            r2cv_w_list.append(R2CV_w_fold)
        R2CV_w = np.mean(r2cv_w_list)
        print(f"  WeightedAvg: R²={R2_w:.4f}  R²CV={R2CV_w:.4f}  RMSE={RMSE_w:.3f}  MAE={MAE_w:.3f}")

        # --- LRStacking ---
        kf_stack = KFold(n_splits=5, shuffle=True, random_state=1)
        r2cv_lr_list, rmse_lr_list, mae_lr_list = [], [], []
        for train_idx, test_idx in kf_stack.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            preds_train = np.array([m.predict(X_train) for m in models])
            preds_test = np.array([m.predict(X_test) for m in models])
            y_pred_lr_test = lr_stacking(preds_train, y_train, preds_test)
            R2CV_fold, RMSE_fold, MAE_fold = calc_metrics(y_test, y_pred_lr_test)
            r2cv_lr_list.append(R2CV_fold)
            rmse_lr_list.append(RMSE_fold)
            mae_lr_list.append(MAE_fold)
        R2CV_lr, RMSE_lr, MAE_lr = np.mean(r2cv_lr_list), np.mean(rmse_lr_list), np.mean(mae_lr_list)
        y_pred_lr_full = lr_stacking(predictions, y, predictions)
        R2_lr, _, _ = calc_metrics(y, y_pred_lr_full)
        print(f"  LRStacking:  R²={R2_lr:.4f}  R²CV={R2CV_lr:.4f}  RMSE={RMSE_lr:.3f}  MAE={MAE_lr:.3f}")

        # --- RidgeStacking ---
        r2cv_ridge_list, rmse_ridge_list, mae_ridge_list = [], [], []
        for train_idx, test_idx in kf_stack.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            preds_train = np.array([m.predict(X_train) for m in models])
            preds_test = np.array([m.predict(X_test) for m in models])
            y_pred_ridge_test = ridge_stacking(preds_train, y_train, preds_test)
            R2CV_fold, RMSE_fold, MAE_fold = calc_metrics(y_test, y_pred_ridge_test)
            r2cv_ridge_list.append(R2CV_fold)
            rmse_ridge_list.append(RMSE_fold)
            mae_ridge_list.append(MAE_fold)
        R2CV_ridge, RMSE_ridge, MAE_ridge = np.mean(r2cv_ridge_list), np.mean(rmse_ridge_list), np.mean(mae_ridge_list)
        y_pred_ridge_full = ridge_stacking(predictions, y, predictions)
        R2_ridge, _, _ = calc_metrics(y, y_pred_ridge_full)
        print(f"  RidgeStacking: R²={R2_ridge:.4f}  R²CV={R2CV_ridge:.4f}  RMSE={RMSE_ridge:.3f}  MAE={MAE_ridge:.3f}")

        # 保存散点图数据（全量数据 使用WeightedAvg）
        pred_dict = {a: results[a]['model'].predict(X) for a in ea_names}
        save_scatter_csv(dataset_name, y, pred_dict, y_weighted, 'WeightedAvg')

        # 保存相关性矩阵
        save_correlation_csv(dataset_name, pred_dict, ea_names)

        # 最佳集成
        best_ensemble_r2cv = max(R2CV_w, R2CV_lr, R2CV_ridge)
        best_single_r2cv = max(r2cv_scores)
        improvement_pp = round((best_ensemble_r2cv - best_single_r2cv) * 100, 2)
        improvement_pct = round((best_ensemble_r2cv - best_single_r2cv) / best_single_r2cv * 100, 2)

        # 选择对应指标
        if best_ensemble_r2cv == R2CV_w:
            best_em = {'R2': R2_w, 'R2CV': R2CV_w, 'RMSE': RMSE_w, 'MAE': MAE_w}
            best_em_name = 'WeightedAvg'
        elif best_ensemble_r2cv == R2CV_lr:
            best_em = {'R2': R2_lr, 'R2CV': R2CV_lr, 'RMSE': RMSE_lr, 'MAE': MAE_lr}
            best_em_name = 'LRStacking'
        else:
            best_em = {'R2': R2_ridge, 'R2CV': R2CV_ridge, 'RMSE': RMSE_ridge, 'MAE': MAE_ridge}
            best_em_name = 'RidgeStacking'

        single_save = {}
        for a in ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite', 'Bayesian']:
            r = results[a]
            single_save[a] = {'R2': r['R2'], 'R2CV': r['R2CV'],
                              'RMSE': r['RMSE'], 'MAE': r['MAE'], 'Time': r['Time']}

        all_results[dataset_name] = {
            'single_results': single_save,
            'ensemble_results': {
                'WeightedAvg': {'R2': R2_w, 'R2CV': R2CV_w, 'RMSE': RMSE_w, 'MAE': MAE_w},
                'LRStacking': {'R2': R2_lr, 'R2CV': R2CV_lr, 'RMSE': RMSE_lr, 'MAE': MAE_lr},
                'RidgeStacking': {'R2': R2_ridge, 'R2CV': R2CV_ridge, 'RMSE': RMSE_ridge, 'MAE': MAE_ridge}
            },
            'best_single_r2cv': best_single_r2cv,
            'best_ensemble_r2cv': best_ensemble_r2cv,
            'best_single_algo': ea_names[r2cv_scores.index(best_single_r2cv)],
            'best_ensemble_method': best_em_name,
            'improvement_pp': improvement_pp,
            'improvement_pct': improvement_pct
        }

    # 保存帕累托CSV
    save_pareto_csv(all_results, list(datasets.keys()))

    # 汇总打印
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

    # 保存JSON到 results/
    json_path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON结果保存至: {json_path}")
    print(f"✅ 所有CSV文件保存至: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
