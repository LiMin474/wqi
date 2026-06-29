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


def SumSqr_BOA(params, XX, YY, cvss, max_iter=300):
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


def a4_BOA_fitrnet_opt(Pred, Resp, max_evals=60):
    """
    Baboon Optimization Algorithm (BOA) for ANN hyperparameter tuning.
    
    Reference:
        Deng (2026). Baboon Optimization Algorithm. Ain Shams Engineering Journal.
    
    Core mechanisms:
        1. Hierarchical population: Leader layer (p1), Adult layer (p2), Juvenile layer (p3)
        2. Foraging phase: Narrowing search (local exploitation) vs Extensive search (global exploration)
        3. Stress response: Population contraction + Random evacuation (Levy flight)
    """
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    # --- BOA parameters ---
    popsize = 15                       # population size
    p1 = 0.15                          # leader layer proportion (top p1)
    p2 = 0.30                          # adult layer proportion
    p3 = 1.0 - p1 - p2                # juvenile layer proportion (= 0.55)
    
    # Foraging phase thresholds
    narrowing_threshold = 0.15         # narrowing step size (local exploitation)
    extensive_scale = 0.4              # extensive search step size (global exploration)
    
    # Stress response probability
    stress_prob = 0.1                  # probability of triggering stress response
    contraction_rate = 0.5             # population contraction strength
    evacuation_rate = 0.6              # random evacuation strength
    
    # Search intensity decay
    decay_rate = 0.95                  # per-generation decay of search intensity

    total_evals = popsize + popsize * ((max_evals - popsize) // popsize)

    print(f'  Running BOA (pop={popsize}, ~{total_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    # --- Initialize population ---
    pop = np.random.rand(popsize, n_params)
    fitness = np.full(popsize, np.inf)
    success_history = np.zeros(popsize, dtype=bool)  # track recent success
    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []
    best_x_global = pop[0].copy()

    # --- Initial evaluation ---
    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_BOA(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            best_x_global = pop[i].copy()
            convergence_history.append((eval_count, best_r2cv))
            print(f'  BOA init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    # Track improvement for foraging phase decision
    last_improvement_evals = 0

    # --- Main loop ---
    gen = 0
    while eval_count < max_evals:
        gen += 1
        search_intensity = decay_rate ** gen  # decay over generations

        # 1. Sort population by fitness
        sorted_idx = np.argsort(fitness)
        n_leader = max(1, int(popsize * p1))
        n_adult = max(1, int(popsize * p2))

        leader_idx = sorted_idx[:n_leader]
        adult_idx = sorted_idx[n_leader:n_leader + n_adult]
        juvenile_idx = sorted_idx[n_leader + n_adult:]

        # Best leader position
        best_pos = pop[leader_idx[0]]

        # Mean position of leaders (for adult guidance)
        leader_mean = np.mean(pop[leader_idx], axis=0)

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            trial = pop[i].copy()

            # --- Determine role-based update ---
            has_improved = (eval_count - last_improvement_evals) < int(0.15 * max_evals)

            if i in leader_idx:
                # --- Leader layer: global exploration ---
                if has_improved:
                    # Narrowing search: explore around current best
                    r = np.random.uniform(0, 1, n_params)
                    trial = best_pos + narrowing_threshold * search_intensity * r * (np.random.rand(n_params) - 0.5) * 2
                else:
                    # Extensive search: explore widely
                    levy_step = levy_flight(n_params)
                    trial = pop[i] + extensive_scale * search_intensity * levy_step

            elif i in adult_idx:
                # --- Adult layer: regional search with information exchange ---
                if has_improved:
                    # Narrowing: move toward leader while maintaining individuality
                    r1 = np.random.rand(n_params)
                    r2 = np.random.rand(n_params)
                    trial = pop[i] + r1 * (best_pos - pop[i]) + r2 * (leader_mean - pop[i])
                else:
                    # Extensive: explore with random perturbation
                    other = np.random.choice([j for j in range(popsize) if j != i])
                    trial = pop[i] + extensive_scale * search_intensity * (pop[other] - pop[i])
                    trial += 0.01 * search_intensity * np.random.randn(n_params)

            else:
                # --- Juvenile layer: fine-grained local exploitation ---
                if has_improved:
                    # Strong attraction to best with small random perturbation
                    r = np.random.rand(n_params)
                    trial = best_pos + 0.1 * r * (pop[i] - best_pos)
                    trial += 0.005 * search_intensity * np.random.randn(n_params)
                else:
                    # Random exploration in vicinity
                    trial = pop[i] + 0.1 * extensive_scale * search_intensity * np.random.randn(n_params)

            # --- Stress response (periodic perturbation) ---
            if np.random.rand() < stress_prob * search_intensity:
                if np.random.rand() < 0.5:
                    # Population contraction: rush toward best
                    trial = best_pos + contraction_rate * np.random.rand(n_params) * (pop[i] - best_pos)
                else:
                    # Random evacuation: Levy flight scattering
                    levy_step = levy_flight(n_params)
                    trial = pop[i] + evacuation_rate * levy_step * search_intensity

            # --- Boundary handling ---
            trial = np.clip(trial, bounds_low, bounds_high)

            # --- Evaluate ---
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_BOA(params_t, Pred, Resp, cvss, max_iter=300)
            eval_count += 1

            # --- Selection (greedy) ---
            if target_t < fitness[i]:
                pop[i] = trial
                fitness[i] = target_t
                success_history[i] = True

                if target_t < best_target:
                    best_target = target_t
                    best_r2cv = output_t['R2CV']
                    best_x_global = trial.copy()
                    last_improvement_evals = eval_count
                    convergence_history.append((eval_count, best_r2cv))
                    print(f'  BOA gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                          f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}',
                          flush=True)
            else:
                success_history[i] = False

            if eval_count >= max_evals:
                break

        # --- Log generation summary ---
        best_idx = np.argmin(fitness)
        print(f'  >>> BOA gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'success_rate={np.mean(success_history):.2%} | '
              f'intensity={search_intensity:.3f}', flush=True)

    # --- Final result ---
    print(f'  BOA complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_BOA(best_params, Pred, Resp, cvss, max_iter=300)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'BOA_evals': eval_count,
        'BOA_convergence': convergence_history
    }

    return Mdl, A1
