import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from common_codes.models import DEFAULT_CONFIG

from common_codes.optimizers.DE import a4_DE_fitrnet_opt
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt


class EnsembleModel:
    def __init__(self, meta, base_models, algo_names):
        self.meta = meta
        self.base_models = base_models
        self.algo_names = algo_names

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([
            self.base_models[name].predict(X) for name in self.algo_names
        ])
        return self.meta.predict(preds)


def _train_model_with_params(params, X, y, model_config):
    """用固定参数直接训练模型（不优化）"""
    _evaluate = model_config['evaluate']
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cvss = list(kf.split(X))
    target, output = _evaluate(params, X, y, cvss)
    return output['Mdl']


def a4_ensemble_stacking(X, y, model_config=None, max_evals=None):
    """
    Stacking ensemble: DE + SHADE + CMA-ES + NRBO + BOA + HHO-Lite

    论文实验设计（严谨且高效）：
    - 外层 5-fold CV 评估最终性能
    - 每个外层 fold 内：
      1. 在 outer_train 上跑 6 个 EA，各得到一组最优超参数（只调一次）
      2. 固定这 6 组参数，在 outer_train 内做 5-fold OOF 预测，训练 meta learner
      3. 用固定参数在完整 outer_train 上重训 6 个基模型
      4. 预测 outer_test，meta learner 输出 stacking 预测
    - 汇总所有 outer_test 预测计算最终 R2CV

    Parameters:
        X, y: 训练数据
        model_config: 模型配置
        max_evals: 每个 EA 的评估预算

    Returns:
        ensemble: EnsembleModel 对象
        A1: dict with R2CV values
    """
    if model_config is None:
        model_config = DEFAULT_CONFIG

    n_folds = 5
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    algo_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    algo_funcs = [a4_DE_fitrnet_opt, a4_SHADE_fitrnet_opt,
                  a4_CMAES_fitrnet_opt, a4_NRBO_fitrnet_opt,
                  a4_BOA_fitrnet_opt, a4_HHO_Lite_fitrnet_opt]

    # --- 外层 CV: 评估最终性能 ---
    print('  === Ensemble: Outer CV for final evaluation ===')
    outer_kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    stacking_cv_preds = np.zeros(n)
    wa_cv_preds = np.zeros(n)
    sa_cv_preds = np.zeros(n)

    for outer_fold_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(X)):
        X_outer_tr = X[outer_train_idx]
        y_outer_tr = y[outer_train_idx]
        X_outer_val = X[outer_val_idx]
        y_outer_val = y[outer_val_idx]
        n_outer_tr = len(outer_train_idx)

        print(f'    Outer fold {outer_fold_idx + 1}/{n_folds}')

        # --- Step 1: 在 outer_train 上跑 EA 调参（只调一次）---
        print('      Step 1: EA optimization on outer_train...')
        outer_tr_models = {}
        outer_tr_best_params = {}
        outer_tr_r2cv = {}

        for name, func in zip(algo_names, algo_funcs):
            kwargs = {'model_config': model_config}
            if max_evals is not None:
                kwargs['max_evals'] = max_evals
            Mdl, A1 = func(X_outer_tr, y_outer_tr, **kwargs)
            outer_tr_models[name] = Mdl
            outer_tr_best_params[name] = A1['best_params']
            outer_tr_r2cv[name] = A1['R2CV']

        # --- Step 2: 固定参数，在 outer_train 内做 5-fold OOF 预测 ---
        print('      Step 2: OOF predictions with fixed params...')
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

        # --- Step 3: 训练 meta learner ---
        print('      Step 3: Train meta learner on OOF...')
        meta = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        meta.fit(inner_oof, y_outer_tr)

        # --- Step 4: 用固定参数在完整 outer_train 上重训基模型 ---
        print('      Step 4: Retrain base models on full outer_train...')
        final_base_models = {}
        for name in algo_names:
            fixed_params = outer_tr_best_params[name]
            final_base_models[name] = _train_model_with_params(fixed_params, X_outer_tr, y_outer_tr, model_config)

        # --- Step 5: 预测 outer_test ---
        outer_val_base_preds = np.column_stack([
            final_base_models[name].predict(X_outer_val) for name in algo_names
        ])
        stacking_cv_preds[outer_val_idx] = meta.predict(outer_val_base_preds).ravel()

        # WeightedAvg 和 SimpleAvg
        r2cv_scores = [outer_tr_r2cv[name] for name in algo_names]
        wa_weights = np.array(r2cv_scores) / (np.sum(r2cv_scores) + 1e-12)
        wa_cv_preds[outer_val_idx] = np.average(outer_val_base_preds, axis=1, weights=wa_weights)
        sa_cv_preds[outer_val_idx] = np.mean(outer_val_base_preds, axis=1)

        print(f'      Outer fold {outer_fold_idx + 1} done')

    # --- 计算最终 R2CV ---
    def calc_r2cv(y_true, y_pred):
        SST = np.sum((y_true - np.mean(y_true)) ** 2)
        SSE = np.sum((y_true - y_pred) ** 2)
        return 1 - SSE / SST if SST != 0 else 0

    stacking_r2cv = float(calc_r2cv(y, stacking_cv_preds))
    wa_r2cv = float(calc_r2cv(y, wa_cv_preds))
    sa_r2cv = float(calc_r2cv(y, sa_cv_preds))

    print(f'    Stacking R2CV = {stacking_r2cv:.4f}')
    print(f'    WeightedAvg R2CV = {wa_r2cv:.4f}')
    print(f'    SimpleAvg R2CV = {sa_r2cv:.4f}')

    # --- 最终模型：在全数据上训练 ---
    print('  === Final: Train ensemble on full data ===')
    final_full_models = {}
    final_full_r2cv = {}
    final_full_params = {}

    for name, func in zip(algo_names, algo_funcs):
        kwargs = {'model_config': model_config}
        if max_evals is not None:
            kwargs['max_evals'] = max_evals
        Mdl, A1 = func(X, y, **kwargs)
        final_full_models[name] = Mdl
        final_full_r2cv[name] = A1['R2CV']
        final_full_params[name] = A1['best_params']

    # OOF predictions on full data
    full_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    full_oof = np.zeros((n, len(algo_names)))

    for fold_idx, (train_idx, val_idx) in enumerate(full_kf.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        for j, name in enumerate(algo_names):
            Mdl_fold = _train_model_with_params(final_full_params[name], X_tr, y_tr, model_config)
            full_oof[val_idx, j] = Mdl_fold.predict(X_va)

    # Final meta learner
    meta_final = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    meta_final.fit(full_oof, y)
    coefs = meta_final.named_steps['lr'].coef_

    # Final predictions
    full_base_preds = np.column_stack([
        final_full_models[name].predict(X) for name in algo_names
    ])
    y_full = meta_final.predict(full_base_preds)
    r2_full = r2_score(y, y_full)

    ensemble = EnsembleModel(meta_final, final_full_models, algo_names)

    A1 = {
        'R2': r2_full,
        'R2CV': stacking_r2cv,
        'WA_R2CV': wa_r2cv,
        'SA_R2CV': sa_r2cv,
        'Model': model_config['name'],
        'algo_names': algo_names,
        'algo_r2cv': final_full_r2cv,
        'meta_coef': coefs.tolist(),
        'meta_intercept': float(meta_final.named_steps['lr'].intercept_),
    }
    return ensemble, A1
