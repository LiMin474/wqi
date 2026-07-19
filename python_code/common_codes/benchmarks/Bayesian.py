import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def SumSqr(params, XX, YY, cvss):
    n_layers = int(params[0])
    layer1 = int(params[1])
    layer2 = int(params[2])
    activation = params[3]
    alpha = float(params[4])

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

    SST = np.sum((YY - np.mean(YY))**2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred)**2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -cv_scores.sum() * len(YY) / len(cv_scores)

    R2 = 1 - SSEmdl / SST
    R2CV = 1 - SSEcv / SST

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def Optopt():
    dimensions = [
        Integer(1, 2, name='NumLayers'),
        Integer(2, 10, name='Layer_1'),
        Integer(2, 10, name='Layer_2'),
        Categorical(['tanh', 'sigmoid', 'relu'], name='Activation'),
        Real(1e-6, 1e-1, 'log-uniform', name='Alpha')
    ]
    return dimensions


def a4_Bayesian_fitrnet_opt(Pred, Resp, max_evals=60):
    numFolds = 5
    np.random.seed(2)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    dimensions = Optopt()

    def objective(params):
        n_layers = int(params[0])
        if n_layers == 1:
            params_list = list(params)
            params_list[2] = 0
            params = tuple(params_list)
        target, _ = SumSqr(params, Pred, Resp, cvss)
        return target

    res = gp_minimize(
        objective,
        dimensions,
        acq_func='EI',
        n_calls=max_evals,
        random_state=2,
        verbose=True
    )

    # Track best-so-far convergence (func_vals = 1-R2CV, convert back to R2CV)
    bayes_best_so_far = 1 - np.minimum.accumulate(res.func_vals)

    print(f'  Best ANN trial found, now computing final model...')
    best_params = res.x
    target, output = SumSqr(best_params, Pred, Resp, cvss)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': int(best_params[0]),
        'Layer_1': int(best_params[1]),
        'Layer_2': int(best_params[2]) if int(best_params[0]) > 1 else 0,
        'Activation': best_params[3],
        'Alpha': float(best_params[4]),
        'R2': output['R2'],
        'Bayesian_convergence': bayes_best_so_far.tolist()
    }

    outer_kf = KFold(n_splits=numFolds, shuffle=True, random_state=42)
    outer_cv_scores = cross_val_score(Mdl, Pred, Resp, cv=outer_kf, scoring='neg_mean_squared_error')
    outer_SSE = -outer_cv_scores.sum() * len(Resp) / numFolds
    outer_SST = np.sum((Resp - np.mean(Resp)) ** 2)
    outer_R2CV = 1 - (outer_SSE / outer_SST) if outer_SST != 0 else 0
    A1['R2CV'] = outer_R2CV
    print(f'  Outer CV R2CV (final report) = {outer_R2CV:.4f}')

    return Mdl, A1
