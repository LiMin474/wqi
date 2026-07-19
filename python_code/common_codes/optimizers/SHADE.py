import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import warnings
warnings.filterwarnings('ignore')
from common_codes.models import DEFAULT_CONFIG


def _decode(x):
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 8))
    layer1 = max(2, min(10, layer1))
    layer2 = int(round(2 + x[2] * 8))
    layer2 = max(2, min(10, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def _evaluate(params, XX, YY, cvss, max_iter=300):
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


def a4_SHADE_fitrnet_opt(Pred, Resp, max_evals=60, model_config=None):
    numFolds = 5
    if model_config is None:
        model_config = DEFAULT_CONFIG
    _decode = model_config['decode']
    _evaluate = model_config['evaluate']
    _get_param_dict = model_config['get_param_dict']
    _param_names = model_config['param_names']
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    popsize = 10
    H = 5
    total_evals = popsize + popsize * ((max_evals - popsize) // popsize)

    print(f'  Running SHADE (pop={popsize}, H={H}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    M_F = np.full(H, 0.5)
    M_Cr = np.full(H, 0.5)
    archive = []
    k = 0

    pop = np.random.rand(popsize, n_params)
    fitness = np.full(popsize, np.inf)

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []

    for i in range(popsize):
        params = _decode(pop[i])
        target, output = _evaluate(params, Pred, Resp, cvss)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            convergence_history.append((eval_count, best_r2cv))
            print(f'  SHADE init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'params={params}', flush=True)

    gen = 0
    while eval_count < max_evals:
        gen += 1
        S_F = []
        S_Cr = []
        S_delta = []

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            ri = np.random.randint(0, H)
            Fi = np.random.normal(loc=M_F[ri], scale=0.1)
            Fi = np.clip(Fi, 0.0, 1.0)
            Cri = np.random.normal(loc=M_Cr[ri], scale=0.1)
            Cri = np.clip(Cri, 0.0, 1.0)

            candidates = [j for j in range(popsize) if j != i]
            p = np.random.choice(candidates)
            pool = candidates + list(range(popsize, popsize + len(archive)))
            if len(archive) > 0:
                idxs = np.random.choice(pool, 2, replace=False)
            else:
                idxs = np.random.choice(candidates, 2, replace=False)

            a = pop[idxs[0]] if idxs[0] < popsize else archive[idxs[0] - popsize]
            b = pop[idxs[1]] if idxs[1] < popsize else archive[idxs[1] - popsize]

            mutant = pop[i] + Fi * (pop[p] - pop[i]) + Fi * (a - b)
            mutant = np.clip(mutant, bounds_low, bounds_high)

            j_rand = np.random.randint(n_params)
            trial = np.where(np.random.rand(n_params) < Cri, mutant, pop[i])
            trial[j_rand] = mutant[j_rand]
            trial = np.clip(trial, bounds_low, bounds_high)

            params_t = _decode(trial)
            target_t, output_t = _evaluate(params_t, Pred, Resp, cvss)
            eval_count += 1

            if target_t < fitness[i]:
                S_F.append(Fi)
                S_Cr.append(Cri)
                S_delta.append(fitness[i] - target_t)
                pop[i] = trial
                fitness[i] = target_t

                if len(archive) >= popsize:
                    del archive[np.random.randint(0, len(archive))]

            if target_t < best_target:
                best_target = target_t
                best_r2cv = output_t['R2CV']
                convergence_history.append((eval_count, best_r2cv))
                print(f'  SHADE gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                      f'params={params_t}', flush=True)

            if eval_count >= max_evals:
                break

        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-12)

            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-12)
            mean_Cr = np.sum(weights * np.array(S_Cr))

            M_F[k] = 0.5 * M_F[k] + 0.5 * mean_F
            M_Cr[k] = 0.5 * M_Cr[k] + 0.5 * mean_Cr
            k = (k + 1) % H

        best_idx = np.argmin(fitness)
        print(f'  >>> SHADE gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'F_mean={np.mean(M_F):.3f}, Cr_mean={np.mean(M_Cr):.3f}', flush=True)

    print(f'  SHADE complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = _decode(best_x)
    target, output = _evaluate(best_params, Pred, Resp, cvss)

    Mdl = output['Mdl']
    A1 = _get_param_dict(best_params)
    A1['R2'] = output['R2']
    A1['R2CV'] = output['R2CV']
    A1['SHADE_evals'] = eval_count
    A1['SHADE_convergence'] = convergence_history

    return Mdl, A1
