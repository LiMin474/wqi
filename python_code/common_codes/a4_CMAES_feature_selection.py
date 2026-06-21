import numpy as np
import cma
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import warnings
warnings.filterwarnings('ignore')

VAR_NAMES = ['pH', 'EC', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']


def decode_params(x):
    feature_mask = np.array([xi > 0.5 for xi in x[:12]])
    n_features = np.sum(feature_mask)
    if n_features == 0:
        feature_mask[0] = True

    act_idx = min(int(x[12] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]

    n_layers = 1 if x[13] < 0.5 else 2
    layer1 = int(round(2 + x[14] * 4))
    layer1 = max(2, min(6, layer1))
    layer2 = int(round(2 + x[15] * 4))
    layer2 = max(2, min(6, layer2))
    alpha = 10.0 ** (-5.0 + x[16] * 4.0)

    return feature_mask, activation, n_layers, layer1, layer2, alpha


def make_pipe(hidden_layer_sizes, activation, alpha, max_iter):
    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
    return Pipeline([
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


def SumSqr_CMAES_FS(params, XX, YY, cvss, max_iter=2000):
    feature_mask, activation, n_layers, layer1, layer2, alpha = params
    n_features = np.sum(feature_mask)

    if n_features == 0:
        return 1.0, {'R2': 0.0, 'R2CV': 0.0, 'Mdl': None, 'n_features': 0}

    XX_sub = XX[:, feature_mask]

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    y_all = YY.values.ravel() if hasattr(YY, 'values') else YY.ravel()
    SST = np.sum((y_all - np.mean(y_all))**2)

    all_preds = np.zeros_like(y_all)
    for train_idx, val_idx in cvss:
        X_tr, X_va = XX_sub[train_idx], XX_sub[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]
        fold_pipe = make_pipe(hidden_layer_sizes, activation, alpha, max_iter)
        fold_pipe.fit(X_tr, y_tr)
        all_preds[val_idx] = fold_pipe.predict(X_va).ravel()

    SSEcv = np.sum((y_all - all_preds)**2)
    R2CV = 1 - (SSEcv / SST)

    final_pipe = make_pipe(hidden_layer_sizes, activation, alpha, max_iter)
    final_pipe.fit(XX_sub, y_all)
    y_pred = final_pipe.predict(XX_sub).ravel()
    SSEmdl = np.sum((y_all - y_pred)**2)
    R2 = 1 - (SSEmdl / SST)

    output = {
        'R2': R2, 'R2CV': R2CV, 'Mdl': final_pipe,
        'feature_mask': feature_mask,
        'n_features': int(n_features)
    }
    target = 1 - R2CV
    return target, output


def a4_CMAES_feature_selection(Pred, Resp):
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 17
    x0 = np.array([0.5] * n_params)
    sigma0 = 0.3

    popsize = 17
    n_generations = 3
    total_evals = (n_generations + 1) * popsize

    print(f'  Running CMA-ES Feature Selection ({n_params}-dim, '
          f'pop={popsize}, gen={n_generations}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=200 for fast search, then final retrain with 2000', flush=True)

    opts = {
        'popsize': popsize,
        'seed': 1,
        'CMA_diagonal': False,
        'bounds': [0.0, 1.0],
        'verbose': -9,
        'maxfevals': total_evals,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    best_nfeat = 12
    convergence_history = []

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            x_clipped = np.clip(x, 0.0, 1.0)
            params = decode_params(x_clipped)
            target, output = SumSqr_CMAES_FS(params, Pred, Resp, cvss, max_iter=200)
            fitnesses.append(target)
            eval_count += 1

            if target < best_target:
                best_target = target
                best_r2cv = output['R2CV']
                best_nfeat = output['n_features']
                convergence_history.append((eval_count, best_r2cv, output['n_features']))
                kept = [VAR_NAMES[j] for j in range(12) if params[0][j]]
                print(f'  CMAES-FS eval {eval_count:3d}: R2CV={output["R2CV"]:.4f} | '
                      f'n={output["n_features"]} | {kept}', flush=True)

        es.tell(solutions, fitnesses)

    print(f'  CMA-ES FS complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_x = es.result.xbest
    best_x = np.clip(best_x, 0.0, 1.0)
    best_params = decode_params(best_x)

    target, output = SumSqr_CMAES_FS(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    feature_mask = output['feature_mask']
    kept_vars = [VAR_NAMES[i] for i in range(12) if feature_mask[i]]
    dropped_vars = [VAR_NAMES[i] for i in range(12) if not feature_mask[i]]

    A1 = {
        'FeatureMask': feature_mask,
        'KeptVars': kept_vars,
        'DroppedVars': dropped_vars,
        'N_Features': output['n_features'],
        'Activation': best_params[1],
        'NumLayers': best_params[2],
        'Layer_1': best_params[3],
        'Layer_2': best_params[4],
        'Alpha': best_params[5],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'CMAES_evals': eval_count,
        'CMAES_convergence': convergence_history
    }

    print(f'\n  Final Results (retrained with max_iter=2000):')
    print(f'    Kept features ({len(kept_vars)}): {kept_vars}')
    if dropped_vars:
        print(f'    Dropped features ({len(dropped_vars)}): {dropped_vars}')
    else:
        print(f'    Dropped features: none')
    print(f'    Architecture: {best_params[2]} layer(s), L1={best_params[3]}, L2={best_params[4]}')
    print(f'    Activation: {best_params[1]}, Alpha={best_params[5]:.6f}')
    print(f'    R2={output["R2"]:.4f}, R2CV={output["R2CV"]:.4f}')

    return Mdl, A1