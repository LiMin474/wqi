"""
统一评估框架：单模型、WeightedAvg、Stacking 在同一套 outer-CV 下评估

论文级评估设计：
- 外层 5-fold CV 评估最终性能
- 每个 outer fold：
  1. 在 outer_train 上跑 6 个 EA，各得到一组最优超参数（只调一次）
  2. 用最佳参数在完整 outer_train 上重训 6 个基模型
  3. 预测 outer_test（单模型、WeightedAvg、Stacking）
- 汇总所有 outer_test 预测，计算最终 R2CV、RMSE、MAE
- WeightedAvg 使用 softmax 归一化权重，避免负值问题
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt


def _train_model_with_params(params, X, y, model_config):
    _evaluate = model_config['evaluate']
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cvss = list(kf.split(X))
    target, output = _evaluate(params, X, y, cvss)
    return output['Mdl']


def _softmax_weights(scores):
    scores = np.array(scores, dtype=float)
    scores = np.clip(scores, 0, None)
    exp_scores = np.exp(scores - np.max(scores))
    weights = exp_scores / np.sum(exp_scores)
    return weights


def unified_outer_cv_evaluation(X, y, model_config, max_evals=None):
    """
    在同一套 outer-CV 下评估单模型、WeightedAvg 和 Stacking

    Returns:
        dict: 包含所有方法的 R2CV, RMSE, MAE
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    n_folds = 5

    algo_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    algo_funcs = [a4_DE_fitrnet_opt, a4_SHADE_fitrnet_opt,
                  a4_CMAES_fitrnet_opt, a4_NRBO_fitrnet_opt,
                  a4_BOA_fitrnet_opt, a4_HHO_Lite_fitrnet_opt]

    outer_kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    single_cv_preds = {name: np.zeros(n) for name in algo_names}
    wa_cv_preds = np.zeros(n)
    sa_cv_preds = np.zeros(n)
    stacking_cv_preds = np.zeros(n)

    for outer_fold_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(X)):
        X_outer_tr = X[outer_train_idx]
        y_outer_tr = y[outer_train_idx]
        X_outer_val = X[outer_val_idx]
        n_outer_tr = len(outer_train_idx)

        print(f'    Outer fold {outer_fold_idx + 1}/{n_folds}')

        print('      Step 1: EA optimization on outer_train...')
        outer_tr_best_params = {}
        outer_tr_r2cv = {}

        for name, func in zip(algo_names, algo_funcs):
            kwargs = {'model_config': model_config}
            if max_evals is not None:
                kwargs['max_evals'] = max_evals
            Mdl, A1 = func(X_outer_tr, y_outer_tr, **kwargs)
            outer_tr_best_params[name] = A1['best_params']
            outer_tr_r2cv[name] = A1['R2CV']

        print('      Step 2: Retrain base models on full outer_train...')
        final_base_models = {}
        for name in algo_names:
            fixed_params = outer_tr_best_params[name]
            final_base_models[name] = _train_model_with_params(fixed_params, X_outer_tr, y_outer_tr, model_config)

        print('      Step 3: Predict outer_test...')
        outer_val_base_preds = np.column_stack([
            final_base_models[name].predict(X_outer_val) for name in algo_names
        ])

        for j, name in enumerate(algo_names):
            single_cv_preds[name][outer_val_idx] = outer_val_base_preds[:, j]

        sa_cv_preds[outer_val_idx] = np.mean(outer_val_base_preds, axis=1)

        r2cv_scores = [outer_tr_r2cv[name] for name in algo_names]
        wa_weights = _softmax_weights(r2cv_scores)
        wa_cv_preds[outer_val_idx] = np.average(outer_val_base_preds, axis=1, weights=wa_weights)

        print('      Step 4: Train Stacking meta learner...')
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_oof = np.zeros((n_outer_tr, len(algo_names)))

        for inner_fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(inner_kf.split(X_outer_tr)):
            X_inner_tr = X_outer_tr[inner_tr_idx]
            y_inner_tr = y_outer_tr[inner_tr_idx]
            X_inner_val = X_outer_tr[inner_val_idx]

            for j, name in enumerate(algo_names):
                fixed_params = outer_tr_best_params[name]
                Mdl_inner = _train_model_with_params(fixed_params, X_inner_tr, y_inner_tr, model_config)
                inner_oof[inner_val_idx, j] = Mdl_inner.predict(X_inner_val)

        meta = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        meta.fit(inner_oof, y_outer_tr)
        stacking_cv_preds[outer_val_idx] = meta.predict(outer_val_base_preds).ravel()

        print(f'      Outer fold {outer_fold_idx + 1} done')

    def calc_metrics(y_true, y_pred):
        SST = np.sum((y_true - np.mean(y_true)) ** 2)
        SSE = np.sum((y_true - y_pred) ** 2)
        R2CV = 1 - SSE / SST if SST != 0 else 0
        RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
        MAE = np.mean(np.abs(y_true - y_pred))
        return {'R2CV': float(R2CV), 'RMSE': float(RMSE), 'MAE': float(MAE)}

    results = {}

    for name in algo_names:
        results[name] = calc_metrics(y, single_cv_preds[name])

    results['SimpleAvg'] = calc_metrics(y, sa_cv_preds)
    results['WeightedAvg'] = calc_metrics(y, wa_cv_preds)
    results['Stacking'] = calc_metrics(y, stacking_cv_preds)

    return results
