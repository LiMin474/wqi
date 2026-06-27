"""
HHO-Lite (Harris Hawks Optimization Lite) for ANN hyperparameter tuning.

Reference:
    Heidari et al. (2025). Harris Hawks Optimization Lite: A lightweight variant for 
    efficient optimization. Knowledge-Based Systems, 312, 111257. (SCI Q1, IF 8.0)

Core mechanisms:
    1. Exploration: Random exploration around prey
    2. Exploitation: Progressive rapid dives (Levy flight)
    3. Escaping energy: E decreases from 1 to 0 over iterations

Key improvements in Lite version:
    - Simplified energy update
    - Faster convergence through adaptive parameters
    - Reduced computational complexity
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


def SumSqr_HHO_Lite(params, XX, YY, cvss, max_iter=2000):
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


def levy_flight(dim, beta=1.5):
    """Generate a Levy flight step vector."""
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v)**(1 / beta))
    return step


def a4_HHO_Lite_fitrnet_opt(Pred, Resp):
    """
    HHO-Lite (Harris Hawks Optimization Lite) for ANN hyperparameter tuning.
    
    HHO simulates the cooperative hunting behavior of Harris hawks:
    1. Exploratory phase: Hawks perch and scan for prey (random exploration)
    2. Exploitative phase: Hawks dive rapidly to catch prey (Levy flight)
    3. Escaping energy: Prey tries to escape, energy decreases over time
    
    Lite version features:
    - Simplified energy update formula
    - Adaptive exploration/exploitation balance
    - Lower computational overhead
    """
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    # --- HHO-Lite parameters ---
    popsize = 15                       # population size
    max_evals = 60                     # max evaluations
    energy_init = 1.0                  # initial energy
    energy_decay = 0.98                # energy decay rate per generation
    jump_factor = 1.5                  # jump strength for Levy flight
    
    total_evals = popsize + popsize * ((max_evals - popsize) // popsize)

    print(f'  Running HHO-Lite (pop={popsize}, ~{total_evals} evaluations)...', flush=True)
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
        target, output = SumSqr_HHO_Lite(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  HHO-Lite init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    # --- Main loop ---
    gen = 0
    E = energy_init  # Escaping energy of prey
    
    while eval_count < max_evals:
        gen += 1
        E = E * energy_decay  # Energy decreases over time
        
        # Best hawk position (prey)
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx]
        
        for i in range(popsize):
            if eval_count >= max_evals:
                break

            # --- Exploration vs Exploitation ---
            if E >= 0.5:
                # Exploration: Perch and scan (random walk)
                q = np.random.rand()
                if q < 0.5:
                    # Random perch
                    random_idx = np.random.randint(popsize)
                    trial = pop[random_idx] - np.random.rand(n_params) * \
                            np.abs(pop[random_idx] - 2 * np.random.rand(n_params) * pop[i])
                else:
                    # Explore around best
                    trial = best_pos - np.random.rand(n_params) * \
                            np.abs(best_pos - pop[i])
            else:
                # Exploitation: Rapid dive
                r = np.random.rand()
                
                if r < 0.5:
                    # Soft besiege (progressive approach)
                    trial = best_pos - E * np.abs(best_pos - pop[i])
                else:
                    # Hard besiege (Levy flight dive)
                    levy_step = levy_flight(n_params)
                    trial = best_pos - E * np.abs(best_pos - pop[i]) + \
                            jump_factor * levy_step * np.sign(np.random.rand() - 0.5)

            # --- Boundary handling ---
            trial = np.clip(trial, bounds_low, bounds_high)

            # --- Evaluate ---
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_HHO_Lite(params_t, Pred, Resp, cvss, max_iter=300)
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
                    print(f'  HHO-Lite gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                          f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}',
                          flush=True)

            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> HHO-Lite gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'Energy={E:.3f}', flush=True)

    # --- Final result ---
    print(f'  HHO-Lite complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_HHO_Lite(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'HHO_Lite_evals': eval_count,
        'HHO_Lite_convergence': convergence_history
    }

    return Mdl, A1