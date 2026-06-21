"""
收集5个算法的收敛曲线数据
保存到 convergence_data.json
"""
import numpy as np
import json
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# 数据集
datasets = {
    '1_jajpur': 'datasets/1_jajpur.npz',
    '2_wqi_dataset': 'datasets/2_wqi_dataset.npz',
    '3_sample_dataset': 'datasets/3_sample_dataset.npz',
    '4_akh_wqi': 'datasets/4_akh_wqi.npz'
}

def decode_params(x):
    n_layers = 1 if x[0] < 0.5 else 2  # 0-0.5: 1层, 0.5-1: 2层
    layer1_size = max(2, int(round(x[1] * 8 + 2)))  # 确保至少2个神经元
    layer2_size = max(2, int(round(x[2] * 8 + 2))) if n_layers == 2 else 0
    activations = ['tanh', 'relu', 'logistic']
    act_idx = min(2, int(x[3] * 3))  # 0-0.33: tanh, 0.33-0.66: relu, 0.66-1: logistic
    act_name = activations[act_idx]
    alpha = x[4] * 0.1
    return n_layers, layer1_size, layer2_size, act_name, alpha

def evaluate_params(params, X, y, cv_splits):
    """评估一组参数，返回R2CV"""
    n_layers, l1, l2, act, alpha = decode_params(params)
    r2cv_list = []

    for train_idx, val_idx in cv_splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if n_layers == 1:
            mlp = MLPRegressor(hidden_layer_sizes=(l1,), activation=act,
                             alpha=alpha, max_iter=1000, random_state=1)
        else:
            mlp = MLPRegressor(hidden_layer_sizes=(l1, l2), activation=act,
                             alpha=alpha, max_iter=1000, random_state=1)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)
        r2cv_list.append(r2_score(y_val, y_pred))

    return np.mean(r2cv_list)

def de_optimizer(X, y, max_gen=100, pop_size=20):
    """差分进化算法"""
    bounds = [(0, 1) for _ in range(5)]
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # 初始化种群
    population = np.random.uniform(0, 1, (pop_size, n_params))
    best_fitness = -np.inf
    best_idx = 0
    history = []

    for gen in range(1, max_gen + 1):
        fitness = []
        for i in range(pop_size):
            f = evaluate_params(population[i], X, y, cv_splits)
            fitness.append(f)
            if f > best_fitness:
                best_fitness = f
                best_idx = i

        # 记录收敛历史（每10代）
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

        # DE变异和选择
        new_population = []
        for i in range(pop_size):
            # 变异
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            mutant = population[r1] + 0.5 * (population[r2] - population[r3])
            mutant = np.clip(mutant, 0, 1)

            # 交叉
            crossover_rate = 0.7
            trial = np.copy(population[i])
            for j in range(n_params):
                if np.random.rand() < crossover_rate:
                    trial[j] = mutant[j]

            # 选择
            trial_f = evaluate_params(trial, X, y, cv_splits)
            if trial_f >= fitness[i]:
                new_population.append(trial)
            else:
                new_population.append(population[i])

        population = np.array(new_population)

    return history, best_fitness

def shade_optimizer(X, y, max_gen=100, pop_size=20):
    """SHADE算法"""
    bounds = [(0, 1) for _ in range(5)]
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # 初始化
    population = np.random.uniform(0, 1, (pop_size, n_params))
    F_archive = np.ones(10) * 0.5
    CR_archive = np.ones(10) * 0.5
    best_fitness = -np.inf
    history = []

    for gen in range(1, max_gen + 1):
        new_population = []
        fitness = []

        for i in range(pop_size):
            # 选择F和CR
            F = np.random.choice(F_archive) if len(F_archive) > 0 else 0.5
            CR = np.random.choice(CR_archive) if len(CR_archive) > 0 else 0.5

            # 变异
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(mutant, 0, 1)

            # 交叉
            j_rand = np.random.randint(n_params)
            trial = np.copy(population[i])
            for j in range(n_params):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # 选择
            trial_f = evaluate_params(trial, X, y, cv_splits)
            current_f = evaluate_params(population[i], X, y, cv_splits)

            if trial_f > current_f:
                new_population.append(trial)
                fitness.append(trial_f)
                if trial_f > best_fitness:
                    best_fitness = trial_f
            else:
                new_population.append(population[i])
                fitness.append(current_f)
                if current_f > best_fitness:
                    best_fitness = current_f

        # 更新历史参数
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

        population = np.array(new_population)

    return history, best_fitness

def cmaes_optimizer(X, y, max_gen=100, pop_size=20):
    """简化CMA-ES"""
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # CMA-ES参数
    mean = np.random.uniform(0, 1, n_params)
    sigma = 0.3
    cov = np.eye(n_params) * 0.1
    best_fitness = -np.inf
    history = []

    for gen in range(1, max_gen + 1):
        # 采样
        samples = np.random.multivariate_normal(mean, cov, pop_size)
        samples = np.clip(samples, 0, 1)

        fitness = [evaluate_params(s, X, y, cv_splits) for s in samples]

        # 更新最好解
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]

        # 更新均值
        weights = np.exp(np.array(fitness) / 0.1)
        weights = weights / weights.sum()
        new_mean = np.sum(samples.T * weights, axis=1)
        mean = new_mean

        # 记录历史
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

    return history, best_fitness

def pso_optimizer(X, y, max_gen=100, pop_size=20):
    """PSO算法"""
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # 初始化
    position = np.random.uniform(0, 1, (pop_size, n_params))
    velocity = np.random.uniform(-0.2, 0.2, (pop_size, n_params))
    personal_best = position.copy()
    personal_best_fitness = np.array([evaluate_params(p, X, y, cv_splits) for p in position])
    global_best = personal_best[np.argmax(personal_best_fitness)]
    global_best_fitness = personal_best_fitness.max()
    best_fitness = global_best_fitness
    history = []

    w, c1, c2 = 0.7, 1.5, 1.5

    for gen in range(1, max_gen + 1):
        for i in range(pop_size):
            r1, r2 = np.random.rand(n_params), np.random.rand(n_params)
            velocity[i] = (w * velocity[i] +
                          c1 * r1 * (personal_best[i] - position[i]) +
                          c2 * r2 * (global_best - position[i]))
            position[i] = np.clip(position[i] + velocity[i], 0, 1)

            f = evaluate_params(position[i], X, y, cv_splits)
            if f > personal_best_fitness[i]:
                personal_best[i] = position[i].copy()
                personal_best_fitness[i] = f

            if f > best_fitness:
                best_fitness = f
                global_best = position[i].copy()

        # 记录历史
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

    return history, best_fitness

def apsm_optimizer(X, y, max_gen=100, pop_size=20):
    """APSM-jSO简化版"""
    bounds = [(0, 1) for _ in range(5)]
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # 初始化
    population = np.random.uniform(0, 1, (pop_size, n_params))
    best_fitness = -np.inf
    history = []

    for gen in range(1, max_gen + 1):
        fitness = [evaluate_params(p, X, y, cv_splits) for p in population]

        for i in range(pop_size):
            if fitness[i] > best_fitness:
                best_fitness = fitness[i]

            # 变异
            best_idx = np.argmax(fitness)
            mutant = population[i] + 0.5 * (population[best_idx] - population[i])
            mutant += np.random.uniform(-0.1, 0.1, n_params)
            mutant = np.clip(mutant, 0, 1)

            # 选择
            mutant_f = evaluate_params(mutant, X, y, cv_splits)
            if mutant_f > fitness[i]:
                population[i] = mutant

        # 记录历史
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

    return history, best_fitness

def nrbo_optimizer(X, y, max_gen=100, pop_size=20):
    """NRBO简化版（基于梯度下降）"""
    n_params = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = list(kf.split(X))

    # 初始化
    position = np.random.uniform(0.2, 0.8, (pop_size, n_params))
    best_fitness = -np.inf
    history = []

    for gen in range(1, max_gen + 1):
        fitness = [evaluate_params(p, X, y, cv_splits) for p in position]

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_pos = position[best_idx].copy()

        # 更新（梯度下降 + 扰动）
        for i in range(pop_size):
            grad = best_pos - position[i]
            lr = 0.1 / (1 + gen * 0.01)
            position[i] = position[i] + lr * grad + np.random.uniform(-0.05, 0.05, n_params)
            position[i] = np.clip(position[i], 0, 1)

        # 记录历史
        if gen % 10 == 0 or gen == 1 or gen == max_gen:
            history.append([gen, best_fitness])

    return history, best_fitness

def main():
    results = {}
    MAX_GEN = 100
    POP_SIZE = 20

    algorithms = {
        'DE': de_optimizer,
        'SHADE': shade_optimizer,
        'CMA-ES': cmaes_optimizer,
        'PSO': pso_optimizer,
        'APSM-jSO': apsm_optimizer,
        'NRBO': nrbo_optimizer
    }

    for ds_name, ds_path in datasets.items():
        print(f"\n{'='*50}")
        print(f"数据集: {ds_name}")
        print('='*50)

        data = np.load(ds_path)
        X, y = data['X'], data['y']
        print(f"  样本: {len(y)}, 特征: {X.shape[1]}")

        results[ds_name] = {}

        for alg_name, alg_func in algorithms.items():
            print(f"  {alg_name}...", end=' ', flush=True)
            start = time.time()

            try:
                history, final_f = alg_func(X, y, MAX_GEN, POP_SIZE)
                elapsed = time.time() - start
                results[ds_name][alg_name] = {
                    'history': history,
                    'final_fitness': final_f,
                    'time': elapsed
                }
                print(f"OK (R2CV={final_f:.4f}, {elapsed:.1f}s)")
            except Exception as e:
                print(f"FAIL: {e}")
                results[ds_name][alg_name] = {'history': [], 'error': str(e)}

    # 保存
    output = 'datasets/results/convergence_data.json'
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n保存到: {output}")

if __name__ == '__main__':
    main()
