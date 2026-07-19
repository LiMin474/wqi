"""
XGBoost — 梯度提升树（非NN基准）
参数空间: 树数、深度、学习率、subsample、lambda
"""
import numpy as np

_XGB_IMPORTED = False
_HAS_GPU = None


def _has_gpu():
    global _HAS_GPU
    if _HAS_GPU is None:
        try:
            import xgboost as xgb
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            temp_x = np.random.rand(10, 3).astype(np.float32)
            temp_y = np.random.rand(10).astype(np.float32)
            test_model = xgb.XGBRegressor(
                n_estimators=1, max_depth=2, verbosity=0,
                tree_method='gpu_hist', device='cuda'
            )
            test_model.fit(temp_x, temp_y)
            _HAS_GPU = True
        except Exception:
            _HAS_GPU = False
    return _HAS_GPU


def _ensure_imports():
    global _XGB_IMPORTED
    if not _XGB_IMPORTED:
        global XGBRegressor, StandardScaler, cross_val_score, Pipeline
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        _XGB_IMPORTED = True


def xgboost_decode(x):
    """Decode [0,1]^5 vector into XGBoost hyperparameters."""
    n_estimators = int(round(50 + x[0] * 450))
    n_estimators = max(50, min(500, n_estimators))
    max_depth = int(round(3 + x[1] * 12))
    max_depth = max(3, min(15, max_depth))
    learning_rate = 10.0 ** (-2.0 + x[2] * 1.477)
    subsample = 0.5 + x[3] * 0.5
    reg_lambda = 10.0 ** (-3.0 + x[4] * 4.0)
    return n_estimators, max_depth, learning_rate, subsample, reg_lambda


def xgboost_evaluate(params, XX, YY, cvss):
    """训练 XGBoost，返回 (target, output_dict)"""
    _ensure_imports()
    n_estimators, max_depth, learning_rate, subsample, reg_lambda = params

    use_gpu = _has_gpu()

    Mdl = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=1,
            verbosity=0,
            tree_method='gpu_hist' if use_gpu else 'auto',
            device='cuda' if use_gpu else 'cpu',
        ))
    ])

    Mdl.fit(XX, YY)

    SST = np.sum((YY - np.mean(YY)) ** 2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred) ** 2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -cv_scores.sum() * len(YY) / len(cv_scores)
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def xgboost_get_param_dict(params):
    """将参数元组转为命名字典"""
    n_estimators, max_depth, learning_rate, subsample, reg_lambda = params
    return {
        'NEstimators': n_estimators,
        'MaxDepth': max_depth,
        'LearningRate': f'{learning_rate:.4f}',
        'Subsample': f'{subsample:.3f}',
        'RegLambda': f'{reg_lambda:.4f}',
    }


XGBOOST_CONFIG = {
    'name': 'XGBoost',
    'n_params': 5,
    'param_names': ['NEstimators', 'MaxDepth', 'LearningRate', 'Subsample', 'RegLambda'],
    'decode': xgboost_decode,
    'evaluate': xgboost_evaluate,
    'get_param_dict': xgboost_get_param_dict,
}
