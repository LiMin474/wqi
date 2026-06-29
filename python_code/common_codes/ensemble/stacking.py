import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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


def a4_ensemble_stacking(X, y):
    """
    Stacking ensemble: DE + SHADE + CMA-ES + NRBO + BOA + HHO-Lite (6个进化算�?

    Level 1: 5-fold CV to generate out-of-fold predictions per base learner.
    Level 2: LinearRegression meta-learner on the 6 predictions.
    """
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
    algo_r2cv = {}

    print('  === Ensemble Level 1: 5-fold base learners ===')
    for j, (name, func) in enumerate(zip(algo_names, algo_funcs)):
        fold_r2_list = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            Mdl, _ = func(X_tr, y_tr)
            yp = Mdl.predict(X_va)
            oof_preds[val_idx, j] = yp
            ssr = np.sum((y_va - yp) ** 2)
            sst = np.sum((y_va - np.mean(y_va)) ** 2)
            fold_r2_list.append(1 - ssr / sst)
        algo_r2cv[name] = float(np.mean(fold_r2_list))
        print(f'    {name:<10s}: fold R2={[f"{r:.4f}" for r in fold_r2_list]}  mean={algo_r2cv[name]:.4f}')

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
    print(f'    Ensemble R2CV = {ensemble_r2cv:.4f}')
    coefs = meta.named_steps['lr'].coef_
    for name, c in zip(algo_names, coefs):
        print(f'      {name} weight = {c:+.4f}')

    # --- Retrain on full data for final R² ---
    print('  === Final: re-train base models on full data ===')
    final_models = {}
    for name, func in zip(algo_names, algo_funcs):
        Mdl, _ = func(X, y)
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
        'NumLayers': -1,
        'Layer_1': -1,
        'Layer_2': -1,
        'Activation': 'Ensemble-Stacking',
        'Alpha': -1,
        'algo_names': algo_names,
        'algo_r2cv': algo_r2cv,
        'meta_coef': coefs.tolist(),
        'meta_intercept': float(meta.named_steps['lr'].intercept_),
    }
    return ensemble, A1
