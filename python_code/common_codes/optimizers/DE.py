import numpy as np
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def decode_params(x):
    """
    Decode [0,1]^5 vector into actual ANN hyperparameters.
    x[0] -> NumLayers:      1 or 2
    x[1] -> Layer_1:        [2, 10]
    x[2] -> Layer_2:        [2, 10]
    x[3] -> Activation:     tanh/sigmoid/relu
    x[4] -> Alpha (L2 reg): [1e-6, 1e-1] log scale
    """
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 8))
    layer1 = max(2, min(10, layer1))
    layer2 = int(round(2 + x[2] * 8))
    layer2 = max(2, min(10, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def SumSqr_DE(params, XX, YY, cvss):
    """
    params = (n_layers, layer1, layer2, activation, alpha)
    """
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
            early_stopping=True
        ))
    ])

    Mdl.fit(XX, YY)

    SST = np.sum((YY - np.mean(YY))**2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred)**2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -cv_scores.sum() * len(YY) / len(cv_scores)
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def a4_DE_fitrnet_opt(Pred, Resp, max_evals=60):
    numFolds = 5
    np.random.seed(7)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds = [(0.0, 1.0)] * n_params

    popsize = 2
    n_init = popsize * n_params
    maxiter = max(1, (max_evals - n_init) // (popsize * n_params))
    eval_count = [0]
    best_target = [float('inf')]
    best_r2cv = [0.0]
    convergence_history = []      # list of (eval_num, R2CV) tuples

    def objective(x):
        params = decode_params(x)
        target, output = SumSqr_DE(params, Pred, Resp, cvss)
        eval_count[0] += 1
        if target < best_target[0]:
            best_target[0] = target
            best_r2cv[0] = output['R2CV']
            convergence_history.append((eval_count[0], best_r2cv[0]))
            print(f'  DE eval {eval_count[0]}: best so far -> '
                  f'R2={output["R2"]:.4f}, R2CV={best_r2cv[0]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}')
        return target

    print(f'  Running Differential Evolution (popsize={popsize}, maxiter={maxiter}, ~{max_evals} evaluations)...')

    res = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=1,
        polish=False,
        disp=False
    )

    print(f'  DE complete: {res.nfev} evaluations, best R2CV={best_r2cv[0]:.4f}')
    print(f'  Computing final model with best params...')
    best_x = res.x
    best_params = decode_params(best_x)
    target, output = SumSqr_DE(best_params, Pred, Resp, cvss)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'DE_nfev': res.nfev,
        'DE_convergence': convergence_history
    }

    return Mdl, A1
