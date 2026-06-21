import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


def SumSqr(params, XX, YY, cvss):
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
        random_state=1,
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


def c2_Bayesopt_ann_for_varselect(X, Y):
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(X))

    dimensions = Optopt()

    def objective(params):
        target, _ = SumSqr(params, X, Y, cvss)
        return target

    res = gp_minimize(
        objective,
        dimensions,
        acq_func='EI',
        n_calls=60,
        random_state=1,
        verbose=False
    )

    best_params = res.x
    target, output = SumSqr(best_params, X, Y, cvss)

    return output['R2'], output['R2CV'], output['Mdl']