"""
INFO (Weighted Mean of Vectors) Optimizer for ANN hyperparameter tuning.

Reference:
    Ahmadianfar et al. (2022). INFO: An Efficient Optimization Algorithm based on Weighted Mean of Vectors.
    Expert Systems with Applications, 198, 116516. (SCI Q1, IF 8.7, 900+ citations)

Core mechanisms:
    1. Updating Rule: mean-based law with convergence acceleration
    2. Vector Combining: combine vectors for promising solutions
    3. Local Search: escape low-accuracy solutions

Key advantage: NOT bio-inspired, uses weighted mean of vectors.
Highly cited (900+), robust to noise, suitable for Ireland dataset.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def decode_params(x):
    """Decode [0,1] vector to ANN hyperparameters."""
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 8))
    layer1 = max(2, min(10, layer1))
    layer2 = int(round(2 + x[2] * 8))
    layer2 = max(2, min(10, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha = 10.0 ** (-6.0 + x[4] * 5.0)
    return n_layers, layer1, layer2, activation, alpha


def SumSqr_INFO(params, XX, YY, cvss, max_iter=2000):
    """Evaluate ANN model and return fitness."""
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


def a4_INFO_fitrnet_opt(Pred, Resp):
    """
    INFO (Weighted Mean of Vectors) Optimizer for ANN hyperparameter tuning.
    
    INFO uses three core procedures:
    1. Updating Rule: generate new vectors using weighted mean
    2. Vector Combining: combine vectors for better solutions
    3. Local Search: escape local optima
    
    Key advantage: math-based (not bio-inspired), 900+ citations, noise-robust.
    """
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    # --- INFO parameters ---
    popsize = 10                       # population size
    max_evals = 60                     # max evaluations
    
    # Updating rule parameters
    delta = 0.5                        # convergence factor
    mu = 0.5                           # weighted mean factor
    
    # Local search parameters
    local_search_prob = 0.1            # probability of local search
    local_search_range = 0.1           # local search step size

    print(f'  Running INFO (pop={popsize}, ~{max_evals} evals)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    # --- Initialize population ---
    pop = np.random.rand(popsize, n_params)
    fitness = np.full(popsize, np.inf)
    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []
    best_x_global = pop[0].copy()

    # --- Initial evaluation ---
    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_INFO(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  INFO init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    # --- Main loop ---
    gen = 0
    
    while eval_count < max_evals:
        gen += 1
        
        # Sort population by fitness
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1]
        third_idx = sorted_idx[2]
        
        best_pos = pop[best_idx]
        second_pos = pop[second_idx]
        third_pos = pop[third_idx]
        
        # Population mean
        pop_mean = np.mean(pop, axis=0)

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            # --- Stage 1: Updating Rule ---
            # Weighted mean of best, second, third, and population mean
            w1 = 0.5                        # weight for best
            w2 = 0.3                        # weight for second
            w3 = 0.2                        # weight for third
            
            # Weighted mean vector
            weighted_mean = w1 * best_pos + w2 * second_pos + w3 * third_pos
            
            # Generate new vector using updating rule
            r1 = np.random.rand(n_params)
            r2 = np.random.rand(n_params)
            
            # Convergence acceleration
            factor = delta * (1 - gen / (max_evals // popsize))  # decreasing factor
            
            trial = pop[i] + factor * r1 * (weighted_mean - pop[i])
            trial += factor * r2 * (best_pos - pop_mean)
            
            # --- Stage 2: Vector Combining ---
            # Combine with random vectors for diversity
            if np.random.rand() < 0.5:
                # Combine with best
                trial = mu * trial + (1 - mu) * best_pos
            else:
                # Combine with random individual
                other = np.random.choice([j for j in range(popsize) if j != i])
                trial = mu * trial + (1 - mu) * pop[other]
            
            # --- Stage 3: Local Search ---
            # Escape local optima with fine-grained search
            if np.random.rand() < local_search_prob:
                # Local search around current position
                perturbation = np.random.randn(n_params) * local_search_range
                trial = trial + perturbation
                
                # Or jump toward best with small step
                if np.random.rand() < 0.5:
                    trial = best_pos + 0.05 * np.random.randn(n_params)

            # --- Boundary handling ---
            trial = np.clip(trial, bounds_low, bounds_high)

            # --- Evaluate ---
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_INFO(params_t, Pred, Resp, cvss, max_iter=300)
            eval_count += 1

            # --- Selection (greedy) ---
            if target_t < fitness[i]:
                pop[i] = trial
                fitness[i] = target_t

                if target_t < best_target:
                    best_target = target_t
                    best_r2cv = output_t['R2CV']
                    best_x_global = trial.copy()
                    convergence_history.append((eval_count, best_r2cv))
                    print(f'  INFO gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                          f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}',
                          flush=True)

            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> INFO gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f}', flush=True)

    # --- Final result ---
    print(f'  INFO complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_INFO(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'INFO_evals': eval_count,
        'INFO_convergence': convergence_history
    }

    return Mdl, A1