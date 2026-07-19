"""
MLP (lbfgs) — 基准模型
参数空间: 层数、神经元数、激活函数、L2正则化
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def mlp_lbfgs_decode(x):
    """Decode [0,1]^5 vector into ANN hyperparameters."""
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 8))
    layer1 = max(2, min(10, layer1))
    layer2 = int(round(2 + x[2] * 8))
    layer2 = max(2, min(10, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def mlp_lbfgs_evaluate(params, XX, YY, cvss):
    """训练 MLP(lbfgs)，返回 (target, output_dict)"""
    n_layers, layer1, layer2, activation, alpha = params

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}

    Mdl = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act_map[activation],
            solver='lbfgs',
            alpha=alpha,
            max_iter=300,
            random_state=1,
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


def mlp_lbfgs_get_param_dict(params):
    """将参数元组转为命名字典"""
    n_layers, layer1, layer2, activation, alpha = params
    return {
        'NumLayers': n_layers,
        'Layer_1': layer1,
        'Layer_2': layer2,
        'Activation': activation,
        'Alpha': alpha,
    }


MLP_LBFGS_CONFIG = {
    'name': 'MLP-lbfgs',
    'n_params': 5,
    'param_names': ['NumLayers', 'Layer_1', 'Layer_2', 'Activation', 'Alpha'],
    'decode': mlp_lbfgs_decode,
    'evaluate': mlp_lbfgs_evaluate,
    'get_param_dict': mlp_lbfgs_get_param_dict,
}