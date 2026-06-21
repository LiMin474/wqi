"""
模糊集成实验（CFE-style）
========================
基于模糊逻辑的动态权重集成方法
参考论文：CFE (Luo et al., IEEE TNNLS)
"""

import numpy as np
import os
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'results', 'all_results_v2.json')
N_FOLDS = 5
RANDOM_STATE = 1

DATASET_KEYS = {
    '4_akh_wqi': '4_akh_wqi',
}

ALGO_MAP = {
    'DE': 'DE',
    'SHADE': 'SHADE',
    'APSM-jSO (2023)': 'APSM-jSO',
    'CMA-ES': 'CMA-ES',
    'Bayesian': 'Bayesian',
}


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    return data['X'], data['y'], str(data['name']), str(data['target_name'])


def build_ann(params):
    n_layers = params['NumLayers']
    layer1 = params['Layer_1']
    layer2 = params['Layer_2']
    activation = params['Activation']
    alpha = params.get('Alpha', 1e-6)
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        alpha = 1e-6

    if n_layers == 1:
        hidden_layer_sizes = (layer1,)
    else:
        hidden_layer_sizes = (layer1, layer2)

    act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
    act = act_map.get(activation, 'relu')

    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=act,
            solver='lbfgs',
            alpha=alpha,
            max_iter=2000,
            random_state=RANDOM_STATE,
            early_stopping=True
        ))
    ])


def evaluate(preds, y):
    """评估预测结果"""
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, rmse, mae


class FuzzyEnsemble:
    """
    模糊集成器（CFE-style）
    - 基于模糊逻辑动态调整基模型权重
    - 考虑模型在困难样本上的性能不确定性
    """
    
    def __init__(self, base_models, eta=1.0):
        self.base_models = base_models  # 列表，每个元素是 (name, model)
        self.eta = eta  # 困难样本判定系数
        self.weights_ = None
        self.tau_err_ = None  # 平均误差
        self.delta_err_ = None  # 误差标准差
    
    def _fuzzy_membership(self, x, a, b, c):
        """三角隶属度函数"""
        if x <= a:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0.0
    
    def _fuzzy_inference(self, error_rate):
        """
        模糊推理：根据模型错误率计算权重
        模糊等级：A(很好), B(好), C(较好), D(中等), E(较差), F(差), G(很差)
        """
        # 输入变量：error_rate [0, 1]
        # 输出变量：weight [0, 1]
        
        # 定义模糊集
        levels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        # 输入隶属度
        mu = []
        # A: [0, 0, 0.1]
        mu.append(self._fuzzy_membership(error_rate, 0, 0, 0.1))
        # B: [0, 0.1, 0.2]
        mu.append(self._fuzzy_membership(error_rate, 0, 0.1, 0.2))
        # C: [0.1, 0.2, 0.3]
        mu.append(self._fuzzy_membership(error_rate, 0.1, 0.2, 0.3))
        # D: [0.2, 0.3, 0.4]
        mu.append(self._fuzzy_membership(error_rate, 0.2, 0.3, 0.4))
        # E: [0.3, 0.4, 0.5]
        mu.append(self._fuzzy_membership(error_rate, 0.3, 0.4, 0.5))
        # F: [0.4, 0.5, 0.7]
        mu.append(self._fuzzy_membership(error_rate, 0.4, 0.5, 0.7))
        # G: [0.5, 0.7, 1.0]
        mu.append(self._fuzzy_membership(error_rate, 0.5, 0.7, 1.0))
        
        # 模糊规则：误差越小，权重越大
        # 输出中心值
        output_centers = [0.95, 0.85, 0.75, 0.60, 0.45, 0.30, 0.15]
        
        # 重心法解模糊
        numerator = sum(mu[i] * output_centers[i] for i in range(7))
        denominator = sum(mu[i] for i in range(7))
        
        if denominator == 0:
            return 0.5  # 默认值
        return numerator / denominator
    
    def fit(self, X, y):
        """训练模糊集成器"""
        n = len(y)
        n_models = len(self.base_models)
        
        # 生成各模型的预测
        predictions = np.zeros((n, n_models))
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        
        # 计算各模型的误差指标
        self.tau_err_ = np.zeros(n_models)  # 平均绝对误差
        self.delta_err_ = np.zeros(n_models)  # 误差标准差
        
        for i in range(n_models):
            errors = np.abs(predictions[:, i] - y)
            self.tau_err_[i] = np.mean(errors)
            self.delta_err_[i] = np.std(errors)
        
        # 识别困难样本
        # 困难样本：误差超过 (tau + eta * delta) 的样本
        hard_samples = np.zeros((n, n_models), dtype=bool)
        for i in range(n_models):
            threshold = self.tau_err_[i] + self.eta * self.delta_err_[i]
            hard_samples[:, i] = np.abs(predictions[:, i] - y) > threshold
        
        # 计算各模型的困难样本错误率
        error_rates = np.zeros(n_models)
        for i in range(n_models):
            if np.any(hard_samples[:, i]):
                # 在困难样本上的错误率
                hard_errors = np.abs(predictions[hard_samples[:, i], i] - y[hard_samples[:, i]])
                error_rates[i] = np.mean(hard_errors) / (np.max(y) - np.min(y))
            else:
                error_rates[i] = 0.1  # 默认较小的错误率
        
        # 使用模糊推理计算权重
        self.weights_ = np.array([self._fuzzy_inference(rate) for rate in error_rates])
        
        # 归一化权重
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        return self
    
    def predict(self, X):
        """进行预测"""
        n = len(X)
        n_models = len(self.base_models)
        
        # 获取各模型预测
        predictions = np.zeros((n, n_models))
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        
        # 加权融合
        final_pred = np.dot(predictions, self.weights_)
        return final_pred


def run_experiment():
    print('Loading JSON results...')
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    for dataset_key in DATASET_KEYS:
        print(f'\n{"=" * 70}')
        print(f'Dataset: {dataset_key}')
        print(f'{"=" * 70}')

        filename = DATASET_KEYS[dataset_key]
        X, y, _, _ = load_dataset(filename)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        # 划分训练集和测试集 (80/20)
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f'n_train={len(train_idx)}, n_test={len(test_idx)}')

        # 获取算法列表
        json_dataset = json_data.get(dataset_key, {})
        algo_list = []
        for json_key, algo_name in ALGO_MAP.items():
            if json_key in json_dataset and 'R2CV' in json_dataset[json_key]:
                algo_list.append((algo_name, json_dataset[json_key]))

        algo_names = [a[0] for a in algo_list]
        print(f'Algorithms: {algo_names}')

        # 训练所有基模型
        print(f'\nTraining base models...')
        base_models = []
        for algo_name, params in algo_list:
            Mdl = build_ann(params)
            Mdl.fit(X_train, y_train)
            base_models.append((algo_name, Mdl))
            print(f'  {algo_name} done')

        # 各算法单独表现
        print(f'\n--- Single Algorithm Performance ---')
        algo_test_r2 = {}
        for name, model in base_models:
            r2, rmse, mae = evaluate(model.predict(X_test), y_test)
            algo_test_r2[name] = r2
            print(f'  {name:<12s}: R2={r2:.4f}')

        best_single_name = max(algo_test_r2, key=algo_test_r2.get)
        best_single_r2 = algo_test_r2[best_single_name]
        print(f'\nBest Single: {best_single_name} R2={best_single_r2:.4f}')

        # ===== 对比实验 =====
        print(f'\n--- Ensemble Methods Comparison ---')
        results = []

        # 1. 简单平均
        predictions = np.column_stack([model.predict(X_test) for name, model in base_models])
        sa_pred = np.mean(predictions, axis=1)
        sa_r2, sa_rmse, sa_mae = evaluate(sa_pred, y_test)
        results.append(('SimpleAvg', sa_r2, sa_rmse, sa_mae, sa_r2 - best_single_r2))
        print(f'  [1] SimpleAvg       R2={sa_r2:.4f}  RMSE={sa_rmse:.4f}  MAE={sa_mae:.4f}  Gain={sa_r2 - best_single_r2:+.4f}')

        # 2. 加权平均（按R2）
        weights = np.array([algo_test_r2[name] for name, _ in base_models])
        weights = weights / weights.sum()
        wa_pred = np.average(predictions, axis=1, weights=weights)
        wa_r2, wa_rmse, wa_mae = evaluate(wa_pred, y_test)
        results.append(('WeightedAvg', wa_r2, wa_rmse, wa_mae, wa_r2 - best_single_r2))
        print(f'  [2] WeightedAvg    R2={wa_r2:.4f}  RMSE={wa_rmse:.4f}  MAE={wa_mae:.4f}  Gain={wa_r2 - best_single_r2:+.4f}')

        # 3. 模糊集成 (CFE-style)
        print(f'\n  [3] FuzzyEnsemble  ... training...')
        fuzzy = FuzzyEnsemble(base_models, eta=1.0)
        fuzzy.fit(X_train, y_train)
        fuzzy_pred = fuzzy.predict(X_test)
        fuzzy_r2, fuzzy_rmse, fuzzy_mae = evaluate(fuzzy_pred, y_test)
        results.append(('FuzzyEnsemble', fuzzy_r2, fuzzy_rmse, fuzzy_mae, fuzzy_r2 - best_single_r2))
        print(f'      FuzzyEnsemble  R2={fuzzy_r2:.4f}  RMSE={fuzzy_rmse:.4f}  MAE={fuzzy_mae:.4f}  Gain={fuzzy_r2 - best_single_r2:+.4f}')

        # 打印模糊权重
        print(f'\n      Fuzzy Weights:')
        for i, (name, _) in enumerate(base_models):
            print(f'        {name:<12s}: {fuzzy.weights_[i]:.4f}')

        # ===== 汇总 =====
        print(f'\n{"=" * 70}')
        print(f'SUMMARY for {dataset_key}')
        print(f'{"=" * 70}')
        print(f'Best Single: {best_single_name} R2={best_single_r2:.4f}')
        print(f'{"-" * 70}')
        print(f'{"Method":<15} {"R2":>10} {"RMSE":>10} {"MAE":>10} {"Gain":>10}')
        print(f'{"-" * 70}')

        results.sort(key=lambda x: x[4], reverse=True)
        for method, r2, rmse, mae, gain in results:
            print(f'{method:<15} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {gain:>+10.4f}')

        best_method = results[0][0]
        best_gain = results[0][4]
        print(f'\nBest Ensemble Method: {best_method} (Gain: {best_gain:+.4f})')


if __name__ == '__main__':
    run_experiment()
