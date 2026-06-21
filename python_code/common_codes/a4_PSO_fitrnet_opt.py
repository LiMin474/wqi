import numpy as np
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


def SumSqr_PSO(params, XX, YY, cvss, max_iter=2000):
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


def a4_PSO_fitrnet_opt(Pred, Resp):
    """
    Particle Swarm Optimization (PSO) with inertia weight.
    Family: Swarm Intelligence — fundamentally different from DE, GA, ES.
    
    Key mechanisms:
    - Each particle has position + velocity
    - Tracks personal best (pbest) and swarm global best (gbest)
    - Inertia weight w linearly decreases from 0.9 to 0.4
    """
    numFolds = 5
    np.random.seed(1)
    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = 0.0
    bounds_high = 1.0

    popsize = 15
    max_evals = 60
    total_evals = popsize + popsize * ((max_evals - popsize) // popsize)

    # PSO parameters
    w_start = 0.9
    w_end = 0.4
    c1 = 2.0  # cognitive coefficient
    c2 = 2.0  # social coefficient
    v_max = 0.2 * (bounds_high - bounds_low)  # max velocity

    print(f'  Running PSO (pop={popsize}, inertia w={w_start}->{w_end}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    # Initialize particles
    pop = np.random.rand(popsize, n_params)
    velocity = np.random.uniform(-v_max, v_max, (popsize, n_params))
    fitness = np.full(popsize, np.inf)

    # Personal bests
    pbest_pos = pop.copy()
    pbest_fit = np.full(popsize, np.inf)

    # Global best
    gbest_pos = pop[0].copy()
    gbest_fit = float('inf')

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []

    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_PSO(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        pbest_fit[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            gbest_fit = target
            gbest_pos = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  PSO init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    gen = 0
    while eval_count < max_evals:
        gen += 1
        w = w_start - (w_start - w_end) * gen / ((max_evals - popsize) // popsize + 1)

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            # Random coefficients
            r1 = np.random.rand(n_params)
            r2 = np.random.rand(n_params)

            # Velocity update
            velocity[i] = (w * velocity[i]
                           + c1 * r1 * (pbest_pos[i] - pop[i])
                           + c2 * r2 * (gbest_pos - pop[i]))
            velocity[i] = np.clip(velocity[i], -v_max, v_max)

            # Position update
            trial = pop[i] + velocity[i]
            trial = np.clip(trial, bounds_low, bounds_high)

            # Evaluate
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_PSO(params_t, Pred, Resp, cvss, max_iter=300)
            eval_count += 1

            if target_t < fitness[i]:
                pop[i] = trial
                fitness[i] = target_t

            # Update personal best
            if target_t < pbest_fit[i]:
                pbest_pos[i] = trial
                pbest_fit[i] = target_t

            # Update global best
            if target_t < gbest_fit:
                gbest_fit = target_t
                gbest_pos = trial.copy()
                best_target = target_t
                best_r2cv = output_t['R2CV']
                convergence_history.append((eval_count, best_r2cv))
                print(f'  PSO gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                      f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}',
                      flush=True)

        best_idx = np.argmin(fitness)
        print(f'  >>> PSO gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | w={w:.3f}', flush=True)

    print(f'  PSO complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_x = gbest_pos
    best_params = decode_params(best_x)
    target, output = SumSqr_PSO(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'PSO_evals': eval_count,
        'PSO_convergence': convergence_history
    }
    return Mdl, A1