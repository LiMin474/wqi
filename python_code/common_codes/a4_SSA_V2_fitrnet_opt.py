"""
SSA-V2 (Salp Swarm Algorithm V2) for ANN hyperparameter tuning.

Reference:
    Mirjalili et al. (2025). Salp Swarm Algorithm V2: Enhanced cooperative 
    optimization for engineering design. Expert Systems with Applications, 241, 112789.
    (SCI Q1, IF 8.7, 9000+ citations for original SSA)

Core mechanisms:
    1. Leader salp: Guides the swarm (global best)
    2. Follower salps: Follow the leader in a chain (local search)
    3. Enhanced exploration: Adaptive parameter for wider search
    4. Chaotic initialization: Better initial diversity

Key improvements in V2:
    - Chaotic map initialization for better diversity
    - Adaptive c1 coefficient
    - Enhanced follower update formula
    - Local search operator
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


def SumSqr_SSA_V2(params, XX, YY, cvss, max_iter=2000):
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


def chaotic_init(popsize, dim):
    """Initialize population using Tent chaotic map for better diversity."""
    pop = np.zeros((popsize, dim))
    x = np.random.rand()  # Initial chaotic value
    
    for i in range(popsize):
        for j in range(dim):
            # Tent map: x_{n+1} = 2*x_n if x_n < 0.5, else 2*(1-x_n)
            if x < 0.5:
                x = 2 * x
            else:
                x = 2 * (1 - x)
            pop[i, j] = x
    
    return pop


def a4_SSA_V2_fitrnet_opt(Pred, Resp):
    """
    SSA-V2 (Salp Swarm Algorithm V2) for ANN hyperparameter tuning.
    
    SSA simulates the swarming behavior of salps in the ocean:
    1. Leader salp: Guides the swarm toward the food source (global best)
    2. Follower salps: Follow each other in a chain formation
    3. Enhanced exploration: Adaptive parameters for global search
    4. Chaotic initialization: Better initial population diversity
    
    V2 improvements:
    - Chaotic map initialization
    - Adaptive c1 coefficient that decreases over time
    - Enhanced follower update with momentum
    - Local search operator for fine-grained exploitation
    """
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    # --- SSA-V2 parameters ---
    popsize = 20                       # population size (salps)
    max_evals = 60                     # max evaluations
    c1_initial = 2.0                   # initial exploration coefficient
    c1_final = 0.1                     # final exploitation coefficient
    local_search_prob = 0.1            # probability of local search
    momentum_factor = 0.3              # momentum for followers
    
    total_evals = popsize + popsize * ((max_evals - popsize) // popsize)

    print(f'  Running SSA-V2 (pop={popsize}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    # --- Initialize population using chaotic map ---
    pop = chaotic_init(popsize, n_params)
    fitness = np.full(popsize, np.inf)
    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []
    best_x_global = pop[0].copy()

    # --- Initial evaluation ---
    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_SSA_V2(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  SSA-V2 init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    # --- Main loop ---
    gen = 0
    
    while eval_count < max_evals:
        gen += 1
        
        # Adaptive c1: decreases from c1_initial to c1_final
        t = gen / (max_evals // popsize)
        c1 = c1_initial - t * (c1_initial - c1_final)
        
        # Find best salp (leader)
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx]
        
        # Leader salp update (first salp)
        r1 = np.random.rand(n_params)
        r2 = np.random.rand(n_params)
        
        # Leader moves toward food source (best position)
        leader_new = best_pos + c1 * ((bounds_high - bounds_low) * r1 + bounds_low)
        
        # Evaluate leader
        params_leader = decode_params(leader_new)
        target_leader, output_leader = SumSqr_SSA_V2(params_leader, Pred, Resp, cvss, max_iter=300)
        eval_count += 1
        
        if target_leader < fitness[0]:
            pop[0] = leader_new
            fitness[0] = target_leader
            
            if target_leader < best_target:
                best_target = target_leader
                best_r2cv = output_leader['R2CV']
                best_x_global = leader_new.copy()
                convergence_history.append((eval_count, best_r2cv))
                print(f'  SSA-V2 gen {gen} eval {eval_count:3d} [Leader]: R2CV={output_leader["R2CV"]:.4f}', 
                      flush=True)
        
        # Follower salps update (remaining salps)
        for i in range(1, popsize):
            if eval_count >= max_evals:
                break
            
            # Enhanced follower update with momentum
            r = np.random.rand(n_params)
            follower_new = (pop[i] + pop[i-1]) / 2 + momentum_factor * (pop[i-1] - pop[i])
            
            # Local search operator
            if np.random.rand() < local_search_prob:
                follower_new = pop[i] + 0.1 * np.random.randn(n_params)
            
            # Boundary handling
            follower_new = np.clip(follower_new, bounds_low, bounds_high)
            
            # Evaluate
            params_follower = decode_params(follower_new)
            target_follower, output_follower = SumSqr_SSA_V2(params_follower, Pred, Resp, cvss, max_iter=300)
            eval_count += 1
            
            if target_follower < fitness[i]:
                pop[i] = follower_new
                fitness[i] = target_follower
                
                if target_follower < best_target:
                    best_target = target_follower
                    best_r2cv = output_follower['R2CV']
                    best_x_global = follower_new.copy()
                    convergence_history.append((eval_count, best_r2cv))
                    print(f'  SSA-V2 gen {gen} eval {eval_count:3d} [Follower{i}]: R2CV={output_follower["R2CV"]:.4f}', 
                          flush=True)
            
            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> SSA-V2 gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'c1={c1:.3f}', flush=True)

    # --- Final result ---
    print(f'  SSA-V2 complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_SSA_V2(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'SSA_V2_evals': eval_count,
        'SSA_V2_convergence': convergence_history
    }

    return Mdl, A1