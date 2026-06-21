import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
import os
import warnings
warnings.filterwarnings('ignore')


def SumSqr(params, XX, YY, cvss, j, var_indices=None):
    rng = np.random.RandomState(j)
    if var_indices is None:
        var_indices = np.ones(XX.shape[1], dtype=bool)

    if np.sum(var_indices) < 1:
        return 1.0, None

    n_layers = int(params[0])
    layer1 = int(params[1])
    layer2 = int(params[2])
    activation = params[3]

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}

    Mdl = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=act_map.get(activation, 'relu'),
        max_iter=2000,
        random_state=j,
        early_stopping=True
    )

    Mdl.fit(XX, YY)

    SST = np.sum((YY - np.mean(YY))**2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred)**2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -np.mean(cv_scores) * len(YY) * 5
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def Optopt():
    dimensions = [
        Integer(1, 2, name='NumLayers'),
        Integer(2, 10, name='Layer_1'),
        Integer(2, 10, name='Layer_2'),
        Categorical(['tanh', 'sigmoid', 'relu'], name='Activation')
    ]
    return dimensions


def c0_FITRNET_bayesian_20_times(X, Y):
    numFolds = 5
    fname = 'Fitrnet_bayesian_predictor_result.npz'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, fname)

    all_tabs = []

    for j in range(20):
        np.random.seed(j)
        kf = KFold(n_splits=numFolds, shuffle=True, random_state=j)
        cvss = list(kf.split(X))

        dimensions = Optopt()

        def objective(params):
            target, _ = SumSqr(params, X, Y, cvss, j)
            return target

        res = gp_minimize(
            objective,
            dimensions,
            acq_func='EI',
            n_calls=60,
            random_state=j,
            verbose=False
        )

        best_params = res.x
        target, output = SumSqr(best_params, X, Y, cvss, j)

        Tab = {
            'NumLayers': int(best_params[0]),
            'Layer_1': int(best_params[1]),
            'Layer_2': int(best_params[2]) if int(best_params[0]) > 1 else 0,
            'Activation': best_params[3],
            'R2': output['R2'],
            'CVR2': output['R2CV']
        }
        all_tabs.append(Tab)

    mytable = np.array(all_tabs)
    np.savez(filepath, mytable=mytable)
    print('')