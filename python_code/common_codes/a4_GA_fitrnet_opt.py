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


def SumSqr_GA(params, XX, YY, cvss, max_iter=2000):
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


def sbx_crossover(parent1, parent2, nc=20):
    """Simulated Binary Crossover (SBX)."""
    dim = len(parent1)
    child1 = np.zeros(dim)
    child2 = np.zeros(dim)
    for i in range(dim):
        if np.random.rand() < 0.9:
            if abs(parent1[i] - parent2[i]) < 1e-12:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
                continue
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (nc + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (nc + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
    return child1, child2


def polynomial_mutation(child, nm=20, bounds_low=0.0, bounds_high=1.0):
    """Polynomial mutation."""
    dim = len(child)
    mutated = child.copy()
    for i in range(dim):
        if np.random.rand() < 1.0 / dim:
            u = np.random.rand()
            if u < 0.5:
                delta = (2 * u) ** (1 / (nm + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (nm + 1))
            mutated[i] = child[i] + delta
    return np.clip(mutated, bounds_low, bounds_high)


def tournament_selection(fitness, k=3):
    """Tournament selection (minimization)."""
    n = len(fitness)
    candidates = np.random.choice(n, k, replace=False)
    best = candidates[np.argmin(fitness[candidates])]
    return best


def a4_GA_fitrnet_opt(Pred, Resp):
    """
    Real-coded Genetic Algorithm with SBX crossover + Polynomial mutation.
    Family: GA (Genetic Algorithm) — completely different from DE/ES/Swarm.
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

    print(f'  Running GA (pop={popsize}, SBX crossover, polynomial mutation, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    pop = np.random.rand(popsize, n_params)
    fitness = np.full(popsize, np.inf)

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []

    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_GA(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1
        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            convergence_history.append((eval_count, best_r2cv))
            print(f'  GA init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    gen = 0
    while eval_count < max_evals:
        gen += 1
        new_pop = np.zeros_like(pop)
        new_fitness = np.full(popsize, np.inf)

        # Elitism: keep best individual
        best_idx = np.argmin(fitness)
        new_pop[0] = pop[best_idx].copy()
        new_fitness[0] = fitness[best_idx]

        idx = 1
        while idx < popsize and eval_count < max_evals:
            # Tournament selection for two parents
            p1_idx = tournament_selection(fitness)
            p2_idx = tournament_selection(fitness)

            # SBX crossover
            c1, c2 = sbx_crossover(pop[p1_idx], pop[p2_idx])

            # Polynomial mutation
            c1 = polynomial_mutation(c1, bounds_low=bounds_low, bounds_high=bounds_high)
            c2 = polynomial_mutation(c2, bounds_low=bounds_low, bounds_high=bounds_high)

            for child in [c1, c2]:
                if idx >= popsize or eval_count >= max_evals:
                    break
                params_c = decode_params(child)
                target_c, output_c = SumSqr_GA(params_c, Pred, Resp, cvss, max_iter=300)
                eval_count += 1
                new_pop[idx] = child
                new_fitness[idx] = target_c
                idx += 1

                if target_c < best_target:
                    best_target = target_c
                    best_r2cv = output_c['R2CV']
                    convergence_history.append((eval_count, best_r2cv))
                    print(f'  GA gen {gen} eval {eval_count:3d}: R2CV={output_c["R2CV"]:.4f} | '
                          f'L1={params_c[1]}, L2={params_c[2]}, Act={params_c[3]}, Alpha={params_c[4]:.6f}',
                          flush=True)

        pop = new_pop
        fitness = new_fitness

        best_idx = np.argmin(fitness)
        print(f'  >>> GA gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f}', flush=True)

    print(f'  GA complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_GA(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'GA_evals': eval_count,
        'GA_convergence': convergence_history
    }
    return Mdl, A1