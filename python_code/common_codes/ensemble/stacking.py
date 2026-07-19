import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from common_codes.models import DEFAULT_CONFIG

# 六个进化算法（不含贝叶斯，贝叶斯是对比方法）
from common_codes.optimizers.DE import a4_DE_fitrnet_opt              # DE (1997)
from common_codes.optimizers.SHADE import a4_SHADE_fitrnet_opt        # SHADE (2013)
from common_codes.optimizers.CMAES import a4_CMAES_fitrnet_opt        # CMA-ES (2006)
from common_codes.optimizers.NRBO import a4_NRBO_fitrnet_opt          # NRBO (2024)
from common_codes.optimizers.BOA import a4_BOA_fitrnet_opt            # BOA (2026)
from common_codes.optimizers.HHO_Lite import a4_HHO_Lite_fitrnet_opt  # HHO-Lite (2025)


class EnsembleModel:
    """Wrapper that holds a meta-learner + 6 base ANNs, exposes .predict()."""
    def __init__(self, meta, base_models, algo_names):
        self.meta = meta
        self.base_models = base_models    # dict: name -> Mdl
        self.algo_names = algo_names

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([
            self.base_models[name].predict(X) for name in self.algo_names
        ])
        return self.meta.predict(preds)


def a4_ensemble_stacking(X, y, model_config=None):
    """
    Stacking ensemble: DE + SHADE + CMA-ES + NRBO + BOA + HHO-Lite (6个进化算法)

    Level 1: 5-fold CV to generate out-of-fold predictions per base learner.
    Level 2: LinearRegression meta-learner on the 6 predictions.

    Returns:
        ensemble: EnsembleModel 对象
        A1: dict with Stacking and WeightedAvg R2CV
    """
    if model_config is None:
        model_config = DEFAULT_CONFIG

    n_folds = 5
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    # 六个进化算法
    algo_names = ['DE', 'SHADE', 'CMA-ES', 'NRBO', 'BOA', 'HHO-Lite']
    algo_funcs = [a4_DE_fitrnet_opt, a4_SHADE_fitrnet_opt,
                  a4_CMAES_fitrnet_opt, a4_NRBO_fitrnet_opt,
                  a4_BOA_fitrnet_opt, a4_HHO_Lite_fitrnet_opt]

    # --- Level 1: out-of-fold predictions ---
    oof_preds = np.zeros((n, len(algo_names)))
    # Track per-fold R2 for each algorithm (for WeightedAvg)
    algo_fold_r2 = {name: [] for name in algo_names}
    # Track per-fold WeightedAvg R2 (corrected, no leakage)
    fold_wa_r2_list = []
    # Track per-fold SimpleAvg R2 (corrected, no leakage)
    fold_sa_r2_list = []

    print('  === Ensemble Level 1: 5-fold base learners ===')
    # Outer: fold, inner: algorithm — so we can compute per-fold WeightedAvg
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        fold_preds = np.zeros((len(val_idx), len(algo_names)))
        fold_r2 = []

        for j, (name, func) in enumerate(zip(algo_names, algo_funcs)):
            Mdl, _ = func(X_tr, y_tr, model_config=model_config)
            yp = Mdl.predict(X_va)
            fold_preds[:, j] = yp
            oof_preds[val_idx, j] = yp

            ssr = np.sum((y_va - yp) ** 2)
            sst = np.sum((y_va - np.mean(y_va)) ** 2)
            fold_r2_val = 1 - ssr / sst if sst != 0 else 0
            fold_r2.append(fold_r2_val)
            algo_fold_r2[name].append(fold_r2_val)

        # --- WeightedAvg within this fold (correct: models trained on train_idx only) ---
        fold_weights = np.array(fold_r2) / (np.sum(fold_r2) + 1e-12)
        y_wa = np.average(fold_preds, axis=1, weights=fold_weights)
        ssr_wa = np.sum((y_va - y_wa) ** 2)
        sst_wa = np.sum((y_va - np.mean(y_va)) ** 2)
        fold_wa_r2 = 1 - ssr_wa / sst_wa if sst_wa != 0 else 0
        fold_wa_r2_list.append(fold_wa_r2)

        # --- SimpleAvg within this fold (correct) ---
        y_sa = np.mean(fold_preds, axis=1)
        ssr_sa = np.sum((y_va - y_sa) ** 2)
        fold_sa_r2 = 1 - ssr_sa / sst_wa if sst_wa != 0 else 0
        fold_sa_r2_list.append(fold_sa_r2)

    # Corrected WeightedAvg R2CV and SimpleAvg R2CV
    wa_r2cv = float(np.mean(fold_wa_r2_list))
    sa_r2cv = float(np.mean(fold_sa_r2_list))

    # Algo R2CV (average across folds)
    algo_r2cv = {}
    for name in algo_names:
        algo_r2cv[name] = float(np.mean(algo_fold_r2[name]))
        print(f'    {name:<10s}: fold R2={[f"{r:.4f}" for r in algo_fold_r2[name]]}  mean={algo_r2cv[name]:.4f}')

    # --- Level 2: meta-learner ---
    print('  === Ensemble Level 2: Meta-learner (LinearRegression) ===')
    meta = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    meta.fit(oof_preds, y)

    y_meta_pred = meta.predict(oof_preds)
    ssr = np.sum((y - y_meta_pred) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    ensemble_r2cv = 1 - ssr / sst
    print(f'    Stacking R2CV = {ensemble_r2cv:.4f}')
    print(f'    WeightedAvg R2CV (corrected) = {wa_r2cv:.4f}')
    print(f'    SimpleAvg R2CV (corrected) = {sa_r2cv:.4f}')
    coefs = meta.named_steps['lr'].coef_
    for name, c in zip(algo_names, coefs):
        print(f'      {name} weight = {c:+.4f}')

    # --- Retrain on full data for final R² ---
    print('  === Final: re-train base models on full data ===')
    final_models = {}
    for name, func in zip(algo_names, algo_funcs):
        Mdl, _ = func(X, y, model_config=model_config)
        final_models[name] = Mdl

    full_preds = np.column_stack([
        final_models[name].predict(X) for name in algo_names
    ])
    y_full = meta.predict(full_preds)
    r2_full = r2_score(y, y_full)

    ensemble = EnsembleModel(meta, final_models, algo_names)

    A1 = {
        'R2': r2_full,
        'R2CV': ensemble_r2cv,
        'WA_R2CV': wa_r2cv,
        'SA_R2CV': sa_r2cv,
        'Model': model_config['name'],
        'algo_names': algo_names,
        'algo_r2cv': algo_r2cv,
        'meta_coef': coefs.tolist(),
        'meta_intercept': float(meta.named_steps['lr'].intercept_),
    }
    return ensemble, A1