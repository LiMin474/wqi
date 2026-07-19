"""
Newton-Raphson-based Optimizer (NRBO) for ANN hyperparameter tuning.

Reference:
    Sowmya et al. (2024). NRBO: Newton-Raphson-based Optimizer.
    Available in pymoo library.

Core mechanisms:
    1. Newton-Raphson Search Rule (NRSR): gradient-like update for fast convergence
    2. Trap Avoidance Operator (TAO): escape local optima
    3. Population-based approach with adaptive search

This is the ONLY math-based algorithm (not bio-inspired) in our ensemble,
providing maximum complementarity with DE/PSO/CMA-ES/Bayesian.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from common_codes.models import DEFAULT_CONFIG


def _decode(x):
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


def _evaluate(params, XX, YY, cvss, max_iter=300):
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


def a4_NRBO_fitrnet_opt(Pred, Resp, max_evals=60, model_config=None):
    """
    Newton-Raphson-based Optimizer (NRBO) for ANN hyperparameter tuning.
    
    NRBO combines:
    - Newton-Raphson Search Rule (NRSR): fast convergence using gradient-like updates
    - Trap Avoidance Operator (TAO): escape local optima with random perturbation
    - Population diversity: maintain multiple search directions
    
    Key advantage: math-based (not bio-inspired), maximum complementarity.
    """
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

    # --- NRBO parameters ---
    popsize = 10                       # population size
    deciding_factor = 0.6              # TAO control parameter
    
    # Newton-Raphson parameters
    delta = 0.5                        # step size factor
    beta = 2.0                         # convergence acceleration factor

    print(f'  Running NRBO (pop={popsize}, deciding_factor={deciding_factor}, ~{max_evals} evals)...', flush=True)
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
        params = _decode(pop[i])
        target, output = _evaluate(params, Pred, Resp, cvss)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  NRBO init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'params={params}', flush=True)

    # --- Main loop ---
    gen = 0
    
    while eval_count < max_evals:
        gen += 1
        
        # Best position
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx]
        
        # Population mean
        pop_mean = np.mean(pop, axis=0)

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            # --- Newton-Raphson Search Rule (NRSR) ---
            # Simulate gradient descent using population information
            r1 = np.random.rand(n_params)
            r2 = np.random.rand(n_params)
            
            # Direction toward best
            direction = best_pos - pop[i]
            
            # Newton-like update: x_new = x - f(x)/f'(x)
            # Approximate gradient using population mean
            gradient_approx = (fitness[i] - best_target) * direction
            
            # Update position
            trial = pop[i] - delta * gradient_approx / (np.linalg.norm(direction) + 1e-10)
            
            # Add acceleration term
            trial += beta * r1 * (best_pos - pop_mean) * r2
            
            # --- Trap Avoidance Operator (TAO) ---
            # Escape local optima with random perturbation
            if np.random.rand() < deciding_factor:
                # Random perturbation around current position
                perturbation = np.random.randn(n_params) * 0.1
                trial = trial + perturbation
                
                # Or jump toward random individual
                if np.random.rand() < 0.5:
                    other = np.random.choice([j for j in range(popsize) if j != i])
                    trial = pop[other] + 0.1 * np.random.randn(n_params)

            # --- Boundary handling ---
            trial = np.clip(trial, bounds_low, bounds_high)

            # --- Evaluate ---
            params_t = _decode(trial)
            target_t, output_t = _evaluate(params_t, Pred, Resp, cvss)
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
                    print(f'  NRBO gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                          f'params={params_t}',
                          flush=True)

            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> NRBO gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f}', flush=True)

    # --- Final result ---
    print(f'  NRBO complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = _decode(best_x)
    target, output = _evaluate(best_params, Pred, Resp, cvss)

    Mdl = output['Mdl']
    A1 = _get_param_dict(best_params)
    A1['R2'] = output['R2']
    A1['R2CV'] = output['R2CV']
    A1['NRBO_evals'] = eval_count
    A1['NRBO_convergence'] = convergence_history

    return Mdl, A1
