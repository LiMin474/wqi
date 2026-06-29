import numpy as np
import cma
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def decode_params(x):
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 8))
    layer1 = max(2, min(10, layer1))
    layer2 = int(round(2 + x[2] * 8))
    layer2 = max(2, min(10, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def SumSqr_CMAES(params, XX, YY, cvss, max_iter=300):
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
            max_iter=max_iter,
            random_state=1,
            n_iter_no_change=10,
            tol=1e-4,
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


def a4_CMAES_fitrnet_opt(Pred, Resp, max_evals=50):
    numFolds = 5
    np.random.seed(7)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    x0 = np.array([0.5] * n_params)
    sigma0 = 0.3

    popsize = 10
    n_generations = max(0, max_evals // popsize - 1)

    total_evals = (n_generations + 1) * popsize

    print(f'  Running CMA-ES (pop={popsize}, gen={n_generations}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    opts = {
        'popsize': popsize,
        'seed': 7,
        'CMA_diagonal': False,
        'bounds': [0.0, 1.0],
        'verbose': -9,
        'maxfevals': total_evals,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            x_clipped = np.clip(x, 0.0, 1.0)
            params = decode_params(x_clipped)
            target, output = SumSqr_CMAES(params, Pred, Resp, cvss, max_iter=300)
            fitnesses.append(target)
            eval_count += 1

            if target < best_target:
                best_target = target
                best_r2cv = output['R2CV']
                convergence_history.append((eval_count, best_r2cv))
                print(f'  CMA-ES eval {eval_count:3d}: R2CV={best_r2cv:.4f} | '
                      f'L1={params[1]}, L2={params[2]}, '
                      f'Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

        es.tell(solutions, fitnesses)

    print(f'  CMA-ES complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_x = es.result.xbest
    best_x = np.clip(best_x, 0.0, 1.0)
    best_params = decode_params(best_x)
    target, output = SumSqr_CMAES(best_params, Pred, Resp, cvss, max_iter=300)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'CMAES_evals': eval_count,
        'CMAES_convergence': convergence_history
    }

    return Mdl, A1
