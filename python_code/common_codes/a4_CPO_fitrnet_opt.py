"""
Crested Porcupine Optimizer (CPO) for ANN hyperparameter tuning.

Reference:
    Abdel-Basset et al. (2024). Crested Porcupine Optimizer: A new nature-inspired metaheuristic.
    Knowledge-Based Systems, 284, 111257. (SCI Q1, IF 8.0)

Core mechanisms:
    1. Four defensive behaviors: sight, sound, odor, physical attack
    2. Cyclic population reduction for convergence
    3. Balance between exploration (sight/sound) and exploitation (odor/attack)

Python package: pip install porcupy
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


def SumSqr_CPO(params, XX, YY, cvss, max_iter=2000):
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


def a4_CPO_fitrnet_opt(Pred, Resp):
    """
    Crested Porcupine Optimizer (CPO) for ANN hyperparameter tuning.
    
    CPO simulates four defensive behaviors of crested porcupines:
    - Sight (exploration): least aggressive, visual detection
    - Sound (exploration): auditory warning, medium range
    - Odor (exploitation): chemical defense, local refinement
    - Physical attack (exploitation): most aggressive, fine-tuning
    
    Uses cyclic population reduction to improve convergence.
    """
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    # --- CPO parameters ---
    popsize = 30                       # initial population size
    max_evals = 60                     # max evaluations
    min_popsize = 5                    # minimum population size
    cycles = 5                         # number of population reduction cycles
    alpha = 0.95                       # reduction rate per cycle
    
    # Defense mechanism probabilities
    p_sight = 0.25                     # sight defense (exploration)
    p_sound = 0.25                     # sound defense (exploration)
    p_odor = 0.25                      # odor defense (exploitation)
    p_attack = 0.25                    # physical attack (exploitation)

    print(f'  Running CPO (pop={popsize}, min_pop={min_popsize}, cycles={cycles}, ~{max_evals} evals)...', flush=True)
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
        target, output = SumSqr_CPO(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  CPO init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    # --- Main loop ---
    gen = 0
    current_popsize = popsize
    
    while eval_count < max_evals:
        gen += 1
        
        # --- Cyclic population reduction ---
        if gen % (max_evals // cycles) == 0 and current_popsize > min_popsize:
            # Remove worst individuals
            sorted_idx = np.argsort(fitness)
            keep_idx = sorted_idx[:current_popsize - int(current_popsize * (1 - alpha))]
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            current_popsize = len(pop)
            print(f'  >>> CPO population reduced to {current_popsize}', flush=True)

        # Best position
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx]

        for i in range(current_popsize):
            if eval_count >= max_evals:
                break

            trial = pop[i].copy()
            
            # --- Select defense mechanism ---
            defense_type = np.random.rand()
            
            if defense_type < p_sight:
                # --- Sight defense (exploration): visual detection ---
                # Move toward best with random perturbation
                r1 = np.random.rand(n_params)
                r2 = np.random.rand(n_params)
                trial = best_pos + r1 * (best_pos - pop[i]) * r2
                
            elif defense_type < p_sight + p_sound:
                # --- Sound defense (exploration): auditory warning ---
                # Global search with random individual
                other = np.random.choice([j for j in range(current_popsize) if j != i])
                trial = pop[i] + np.random.rand(n_params) * (pop[other] - pop[i])
                trial += 0.1 * np.random.randn(n_params)
                
            elif defense_type < p_sight + p_sound + p_odor:
                # --- Odor defense (exploitation): chemical defense ---
                # Local refinement around best
                r = np.random.rand(n_params)
                trial = best_pos + 0.1 * r * (pop[i] - best_pos)
                trial += 0.01 * np.random.randn(n_params)
                
            else:
                # --- Physical attack (exploitation): fine-tuning ---
                # Strong convergence toward best
                beta = 2.0 * (1 - gen / (max_evals // popsize))  # decreasing factor
                trial = best_pos + beta * np.random.rand(n_params) * (best_pos - pop[i])

            # --- Boundary handling ---
            trial = np.clip(trial, bounds_low, bounds_high)

            # --- Evaluate ---
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_CPO(params_t, Pred, Resp, cvss, max_iter=300)
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
                    print(f'  CPO gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                          f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}',
                          flush=True)

            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> CPO gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'pop_size={current_popsize}', flush=True)

    # --- Final result ---
    print(f'  CPO complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_CPO(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'CPO_evals': eval_count,
        'CPO_convergence': convergence_history
    }

    return Mdl, A1