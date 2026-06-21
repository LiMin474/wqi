"""
APSM-jSO: jSO with Adaptive Parameter Selection Mechanism + FIFO Archive + RSP Mutation
Li et al., "APSM-jSO: A novel jSO variant with an adaptive parameter selection mechanism
and a new external archive updating mechanism", Swarm and Evolutionary Computation, 2023

Three improvements over jSO:
1. APSM — adaptive selection of memory entries (weighted by success rate, not random)
2. FIFO — first-in-first-out archive replacement
3. RSP — rank-based selective pressure in mutation (fitter individuals selected more often)
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
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


def SumSqr_APSM_jSO(params, XX, YY, cvss, max_iter=2000):
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


def a4_APSM_jSO_fitrnet_opt(Pred, Resp):
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 5
    bounds_low = np.zeros(n_params)
    bounds_high = np.ones(n_params)

    popsize_init = 15
    popsize_min = 4
    H = 5
    max_evals = 60
    max_p = 0.25

    print(f'  Running APSM-jSO (pop_init={popsize_init}, pop_min={popsize_min}, H={H}, ~{max_evals} evaluations)...', flush=True)
    print(f'  Using max_iter=300 for fast search, then final retrain with 2000', flush=True)

    M_F = np.full(H, 0.5)
    M_Cr = np.full(H, 0.5)

    # --- APSM: success counts for each memory entry ---
    success_counts = np.ones(H, dtype=float)  # start uniform

    # --- FIFO archive ---
    archive = []

    k = 0
    popsize = popsize_init
    pop = np.random.rand(popsize, n_params)
    fitness = np.full(popsize, np.inf)

    eval_count = 0
    best_target = float('inf')
    best_r2cv = 0.0
    convergence_history = []

    # Initial evaluation
    for i in range(popsize):
        params = decode_params(pop[i])
        target, output = SumSqr_APSM_jSO(params, Pred, Resp, cvss, max_iter=300)
        fitness[i] = target
        eval_count += 1

        if target < best_target:
            best_target = target
            best_r2cv = output['R2CV']
            convergence_history.append((eval_count, best_r2cv))
            print(f'  APSM-jSO init {i+1:2d}/{popsize}: R2CV={output["R2CV"]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, Alpha={params[4]:.6f}', flush=True)

    gen = 0
    while eval_count < max_evals:
        gen += 1
        S_F = []
        S_Cr = []
        S_delta = []

        sorted_idx = np.argsort(fitness)

        for i in range(popsize):
            if eval_count >= max_evals:
                break

            # --- jSO phase-based F_w ---
            ratio = eval_count / max_evals
            if ratio < 0.2:
                F_w_scale = 0.7
            elif ratio < 0.4:
                F_w_scale = 0.8
            else:
                F_w_scale = 1.2

            # --- APSM: select memory entry weighted by success_counts ---
            probs = success_counts / (success_counts.sum() + 1e-12)
            ri = np.random.choice(H, p=probs)

            Fi = np.random.normal(loc=M_F[ri], scale=0.1)
            Fi = np.clip(Fi, 0.0, 1.0)
            Cri = np.random.normal(loc=M_Cr[ri], scale=0.1)
            Cri = np.clip(Cri, 0.0, 1.0)

            # pbest rate (same as jSO)
            p_rate = max_p * (1.0 - ratio) + 0.05
            n_pbest = max(1, int(popsize * p_rate))
            pbest_idx = sorted_idx[np.random.randint(0, n_pbest)]

            # --- RSP: rank-based selection for r1 and r2 ---
            # Higher rank (better fitness) → higher selection probability
            # We select from candidates using rank-based probabilities
            candidates = [j for j in range(popsize) if j != i]
            n_cand = len(candidates)

            # Rank candidates by fitness (0 = best, n_cand-1 = worst)
            cand_fitness = np.array([fitness[j] for j in candidates])
            cand_ranks = np.argsort(np.argsort(cand_fitness))  # 0 = best, n_cand-1 = worst

            # RSP probability: linearly decreasing with rank
            # P(select rank k) = (n_cand - k) / sum(1..n_cand) = 2*(n_cand-k) / (n_cand*(n_cand+1))
            rsp_probs = (n_cand - cand_ranks) / (n_cand * (n_cand + 1) / 2)
            rsp_probs = rsp_probs / (rsp_probs.sum() + 1e-12)

            r1 = np.random.choice(candidates, p=rsp_probs)

            # For r2, select from pop ∪ archive with RSP
            if len(archive) > 0:
                pool = candidates + list(range(popsize, popsize + len(archive)))
                pool_fitness = np.array(
                    [fitness[j] for j in candidates] +
                    [np.inf] * len(archive)
                )
                pool_ranks = np.argsort(np.argsort(pool_fitness))
                n_pool = len(pool)
                pool_probs = (n_pool - pool_ranks) / (n_pool * (n_pool + 1) / 2)
                pool_probs = pool_probs / (pool_probs.sum() + 1e-12)
                r2 = np.random.choice(pool, p=pool_probs)
            else:
                candidates_no_r1 = [j for j in candidates if j != r1]
                if len(candidates_no_r1) > 1:
                    cand2_fitness = np.array([fitness[j] for j in candidates_no_r1])
                    cand2_ranks = np.argsort(np.argsort(cand2_fitness))
                    n2 = len(candidates_no_r1)
                    probs2 = (n2 - cand2_ranks) / (n2 * (n2 + 1) / 2)
                    probs2 = probs2 / (probs2.sum() + 1e-12)
                    r2 = np.random.choice(candidates_no_r1, p=probs2)
                else:
                    r2 = np.random.choice(candidates_no_r1)

            r2_idx = r2 if r2 < popsize else archive[r2 - popsize]

            # DE/current-to-pbest-w/1 mutation
            F_w = Fi * F_w_scale
            mutant = pop[i] + F_w * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - r2_idx)
            mutant = np.clip(mutant, bounds_low, bounds_high)

            # Crossover
            j_rand = np.random.randint(n_params)
            trial = np.where(np.random.rand(n_params) < Cri, mutant, pop[i])
            trial[j_rand] = mutant[j_rand]
            trial = np.clip(trial, bounds_low, bounds_high)

            # Evaluate
            params_t = decode_params(trial)
            target_t, output_t = SumSqr_APSM_jSO(params_t, Pred, Resp, cvss, max_iter=300)
            eval_count += 1

            # Selection
            if target_t < fitness[i]:
                S_F.append(Fi)
                S_Cr.append(Cri)
                S_delta.append(fitness[i] - target_t)

                # --- FIFO archive: append to end, remove from front if full ---
                archive.append(pop[i].copy())
                if len(archive) > popsize:
                    archive.pop(0)  # FIFO: remove oldest (front)

                pop[i] = trial
                fitness[i] = target_t

                # --- APSM: update success count for chosen memory entry ---
                success_counts[ri] += 1.0

            if target_t < best_target:
                best_target = target_t
                best_r2cv = output_t['R2CV']
                convergence_history.append((eval_count, best_r2cv))
                print(f'  APSM-jSO gen {gen} eval {eval_count:3d}: R2CV={output_t["R2CV"]:.4f} | '
                      f'L1={params_t[1]}, L2={params_t[2]}, Act={params_t[3]}, Alpha={params_t[4]:.6f}', flush=True)

            if eval_count >= max_evals:
                break

        # Update memory (same as jSO/L-SHADE)
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-12)

            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-12)
            mean_Cr = np.sum(weights * np.array(S_Cr))

            M_F[k] = (1 - 0.5) * M_F[k] + 0.5 * mean_F
            M_Cr[k] = (1 - 0.5) * M_Cr[k] + 0.5 * mean_Cr
            k = (k + 1) % H

            # --- APSM: decay success counts to avoid unbounded growth ---
            success_counts *= 0.95
            success_counts = np.clip(success_counts, 1.0, None)

        # Linear population size reduction
        new_popsize = max(popsize_min, int(round(
            popsize_init - (popsize_init - popsize_min) * eval_count / max_evals
        )))
        if new_popsize < popsize:
            n_remove = popsize - new_popsize
            worst_idx = np.argsort(-fitness)[:n_remove]
            pop = np.delete(pop, worst_idx, axis=0)
            fitness = np.delete(fitness, worst_idx)
            popsize = new_popsize

        best_idx = np.argmin(fitness)
        print(f'  >>> APSM-jSO gen {gen:2d}: best R2CV={1-fitness[best_idx]:.4f} | '
              f'pop={popsize}, F_w={F_w_scale:.1f}, p_rate={p_rate:.3f}', flush=True)

    print(f'  APSM-jSO complete: {eval_count} evaluations, best R2CV={best_r2cv:.4f}', flush=True)

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    best_params = decode_params(best_x)
    target, output = SumSqr_APSM_jSO(best_params, Pred, Resp, cvss, max_iter=2000)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha': best_params[4],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'APSM_jSO_evals': eval_count,
        'APSM_jSO_convergence': convergence_history
    }

    return Mdl, A1