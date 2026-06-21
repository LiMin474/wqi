import numpy as np
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
            early_stopping=True
        ))
    ])


def SumSqr_DE_FS(params, XX, YY, cvss, max_iter=2000):
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


def a4_DE_feature_selection(Pred, Resp):
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 17
    pop_size = n_params
    max_gen = 3
    F = 0.8
    Cr = 0.7

    total_evals = pop_size + max_gen * pop_size
    print(f'  Running Manual DE Feature Selection ({n_params}-dim, '
          f'pop={pop_size}, gen={max_gen}, ~{total_evals} evaluations)...')
    print(f'  Using max_iter=200 for fast search, then final retrain with 2000')

    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    pop = np.random.rand(pop_size, n_params)
    fitness = np.full(pop_size, np.inf)
    outputs = [None] * pop_size

    eval_count = 0
    best_f = np.inf
    best_r2cv = 0.0
    best_nfeat = 12
    convergence = []

    def clip_pop(x):
        return np.clip(x, bounds_low, bounds_high)

    for i in range(pop_size):
        params = decode_params(pop[i])
        target, output = SumSqr_DE_FS(params, Pred, Resp, cvss, max_iter=200)
        fitness[i] = target
        outputs[i] = output
        eval_count += 1

        if target < best_f:
            best_f = target
            best_r2cv = output['R2CV']
            best_nfeat = output['n_features']
            convergence.append((eval_count, best_r2cv, output['n_features']))
            kept = [VAR_NAMES[j] for j in range(12) if params[0][j]]
            print(f'  Init {i+1:2d}/{pop_size}: R2CV={output["R2CV"]:.4f} | '
                  f'n={output["n_features"]} | {kept}')
            sys.stdout.flush()

    for gen in range(max_gen):
        gen_best = np.inf
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)
            mutant = clip_pop(mutant)

            trial = np.where(np.random.rand(n_params) < Cr, mutant, pop[i])
            j_rand = np.random.randint(n_params)
            trial[j_rand] = mutant[j_rand]
            trial = clip_pop(trial)

            params_t = decode_params(trial)
            target_t, output_t = SumSqr_DE_FS(params_t, Pred, Resp, cvss, max_iter=200)
            eval_count += 1

            if target_t < fitness[i]:
                pop[i] = trial
                fitness[i] = target_t
                outputs[i] = output_t

            if target_t < best_f:
                best_f = target_t
                best_r2cv = output_t['R2CV']
                best_nfeat = output_t['n_features']
                convergence.append((eval_count, best_r2cv, output_t['n_features']))
                kept = [VAR_NAMES[j] for j in range(12) if params_t[0][j]]
                print(f'  Gen {gen+1}/{max_gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                      f'n={output_t["n_features"]} | {kept}')
                sys.stdout.flush()

            if target_t < gen_best:
                gen_best = target_t

        best_idx = np.argmin(fitness)
        print(f'  >>> Gen {gen+1} done: best R2CV={1-fitness[best_idx]:.4f} '
              f'(n_feat={outputs[best_idx]["n_features"]})')
        sys.stdout.flush()

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)

    print(f'\n  DE search complete: {eval_count} evaluations')
    print(f'  Best found: R2CV={best_r2cv:.4f} with {best_nfeat} features')

    target, output = SumSqr_DE_FS(best_params, Pred, Resp, cvss, max_iter=2000)

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
        'DE_nfev': eval_count,
        'DE_convergence': convergence
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