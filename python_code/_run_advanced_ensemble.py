"""
前沿集成方法实验
================
测试 DES、Ensemble Distillation、MoE 等前沿方法
"""

import numpy as np
import os
import warnings
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
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


def get_oof_preds(X, y, params):
    """为单个算法生成OOF预测"""
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(n)

    for train_idx, val_idx in kf.split(X):
        Mdl = build_ann(params)
        Mdl.fit(X[train_idx], y[train_idx])
        oof[val_idx] = Mdl.predict(X[val_idx])

    return oof


def evaluate(preds, y):
    """评估预测结果"""
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, rmse, mae


class DESRegressor(BaseEstimator, RegressorMixin):
    """
    Dynamic Ensemble Selection (DES)
    根据样本的特征动态选择最优的基模型子集
    """
    def __init__(self, base_models, k=3):
        self.base_models = base_models  # 列表，每个元素是 (name, model)
        self.k = k  # 选择前k个最优模型
        self.meta_model = None
    
    def fit(self, X, y):
        # 为每个基模型生成预测
        n = len(y)
        predictions = np.zeros((n, len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        
        # 训练一个分类器来决定使用哪个模型
        # 这里简化：选择每个样本上预测最准的k个模型
        self.train_predictions_ = predictions
        self.train_y_ = y
        return self
    
    def predict(self, X):
        n = len(X)
        predictions = np.zeros((n, len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        
        # 动态选择：对每个样本，选择在训练集上相关性最高的模型
        final_pred = np.zeros(n)
        
        for i in range(n):
            # 计算每个模型在该样本附近的表现
            # 简化：用训练集的误差来加权
            errors = []
            for j in range(len(self.base_models)):
                # 计算该模型在训练集上的误差
                train_err = np.mean((self.train_predictions_[:, j] - self.train_y_) ** 2)
                errors.append(train_err)
            
            # 选择误差最小的k个模型
            best_idx = np.argsort(errors)[:self.k]
            final_pred[i] = np.mean(predictions[i, best_idx])
        
        return final_pred


class EnsembleDistillation(BaseEstimator, RegressorMixin):
    """
    Ensemble Distillation
    用一个学生模型学习多个老师模型的输出
    """
    def __init__(self, teacher_models, student_model=None):
        self.teacher_models = teacher_models
        if student_model is None:
            self.student_model = MLPRegressor(
                hidden_layer_sizes=(10,),
                activation='relu',
                solver='lbfgs',
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        else:
            self.student_model = student_model
    
    def fit(self, X, y):
        # 生成老师模型的预测作为软标签
        teacher_preds = np.column_stack([
            model.predict(X) for name, model in self.teacher_models
        ])
        
        # 计算老师模型的平均预测作为蒸馏目标
        distilled_target = np.mean(teacher_preds, axis=1)
        
        # 训练学生模型学习蒸馏目标
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.student_model.fit(X_scaled, distilled_target)
        self.scaler_ = scaler
        return self
    
    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        return self.student_model.predict(X_scaled)


class MixtureOfExperts(BaseEstimator, RegressorMixin):
    """
    Mixture of Experts (MoE)
    门控网络决定每个样本由哪个专家处理
    """
    def __init__(self, expert_models):
        self.expert_models = expert_models  # 列表，每个元素是 (name, model)
        self.gating_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=RANDOM_STATE)
    
    def fit(self, X, y):
        # 训练所有专家模型
        for name, model in self.expert_models:
            model.fit(X, y)
        
        # 训练门控模型：预测每个样本应该用哪个专家
        # 简单版本：用专家预测的误差作为标签
        expert_preds = np.column_stack([
            model.predict(X) for name, model in self.expert_models
        ])
        
        # 找到每个样本预测最准的专家
        errors = np.abs(expert_preds - y.reshape(-1, 1))
        best_expert_idx = np.argmin(errors, axis=1)
        
        # 训练门控网络学习特征到专家的映射
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.gating_model.fit(X_scaled, best_expert_idx)
        self.scaler_ = scaler
        return self
    
    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        
        # 门控网络预测每个样本使用哪个专家
        expert_idx = self.gating_model.predict(X_scaled)
        
        # 使用相应的专家进行预测
        n = len(X)
        predictions = np.zeros(n)
        
        for i in range(n):
            _, model = self.expert_models[expert_idx[i]]
            predictions[i] = model.predict(X[i:i+1])[0]
        
        return predictions


class BayesianModelAveraging(BaseEstimator, RegressorMixin):
    """
    Bayesian Model Averaging (BMA)
    按模型后验概率加权平均
    """
    def __init__(self, base_models):
        self.base_models = base_models
        self.weights_ = None
    
    def fit(self, X, y):
        # 简单版本：用模型在训练集上的表现作为权重
        weights = []
        
        for name, model in self.base_models:
            preds = model.predict(X)
            r2 = r2_score(y, preds)
            weights.append(max(r2, 0))  # 避免负权重
        
        # 归一化权重
        weights = np.array(weights)
        self.weights_ = weights / weights.sum()
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for name, model in self.base_models
        ])
        return np.dot(predictions, self.weights_)


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

        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f'n_train={len(train_idx)}, n_test={len(test_idx)}')

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

        # ===== 前沿集成方法对比 =====
        print(f'\n--- Advanced Ensemble Methods ---')
        results = []

        # 1. Dynamic Ensemble Selection (DES)
        des = DESRegressor(base_models, k=3)
        des.fit(X_train, y_train)
        des_pred = des.predict(X_test)
        des_r2, des_rmse, des_mae = evaluate(des_pred, y_test)
        results.append(('DES', des_r2, des_rmse, des_mae, des_r2 - best_single_r2))
        print(f'  [1] DES (k=3)      R2={des_r2:.4f}  RMSE={des_rmse:.4f}  MAE={des_mae:.4f}  Gain={des_r2 - best_single_r2:+.4f}')

        # 2. Ensemble Distillation
        ed = EnsembleDistillation(base_models)
        ed.fit(X_train, y_train)
        ed_pred = ed.predict(X_test)
        ed_r2, ed_rmse, ed_mae = evaluate(ed_pred, y_test)
        results.append(('EnsembleDistillation', ed_r2, ed_rmse, ed_mae, ed_r2 - best_single_r2))
        print(f'  [2] EnsembleDistill R2={ed_r2:.4f}  RMSE={ed_rmse:.4f}  MAE={ed_mae:.4f}  Gain={ed_r2 - best_single_r2:+.4f}')

        # 3. Mixture of Experts (MoE)
        moe = MixtureOfExperts(base_models)
        moe.fit(X_train, y_train)
        moe_pred = moe.predict(X_test)
        moe_r2, moe_rmse, moe_mae = evaluate(moe_pred, y_test)
        results.append(('MoE', moe_r2, moe_rmse, moe_mae, moe_r2 - best_single_r2))
        print(f'  [3] MoE            R2={moe_r2:.4f}  RMSE={moe_rmse:.4f}  MAE={moe_mae:.4f}  Gain={moe_r2 - best_single_r2:+.4f}')

        # 4. Bayesian Model Averaging (BMA)
        bma = BayesianModelAveraging(base_models)
        bma.fit(X_train, y_train)
        bma_pred = bma.predict(X_test)
        bma_r2, bma_rmse, bma_mae = evaluate(bma_pred, y_test)
        results.append(('BMA', bma_r2, bma_rmse, bma_mae, bma_r2 - best_single_r2))
        print(f'  [4] BMA            R2={bma_r2:.4f}  RMSE={bma_rmse:.4f}  MAE={bma_mae:.4f}  Gain={bma_r2 - best_single_r2:+.4f}')

        # 5. 对比：LR Stacking（基准）
        # 生成OOF预测
        oof_train = {}
        for algo_name, params in algo_list:
            oof_train[algo_name] = get_oof_preds(X_train, y_train, params)
        
        test_preds = {}
        for name, model in base_models:
            test_preds[name] = model.predict(X_test)
        
        X_train_stack = np.column_stack([oof_train[a] for a in algo_names])
        X_test_stack = np.column_stack([test_preds[a] for a in algo_names])
        
        lr_meta = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        lr_meta.fit(X_train_stack, y_train)
        lr_pred = lr_meta.predict(X_test_stack)
        lr_r2, lr_rmse, lr_mae = evaluate(lr_pred, y_test)
        results.append(('LRStacking', lr_r2, lr_rmse, lr_mae, lr_r2 - best_single_r2))
        print(f'  [5] LRStacking     R2={lr_r2:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  Gain={lr_r2 - best_single_r2:+.4f}')

        # ===== 汇总 =====
        print(f'\n{"=" * 70}')
        print(f'SUMMARY for {dataset_key}')
        print(f'{"=" * 70}')
        print(f'Best Single: {best_single_name} R2={best_single_r2:.4f}')
        print(f'{"-" * 70}')
        print(f'{"Method":<20} {"R2":>10} {"RMSE":>10} {"MAE":>10} {"Gain":>10}')
        print(f'{"-" * 70}')

        results.sort(key=lambda x: x[4], reverse=True)
        for method, r2, rmse, mae, gain in results:
            print(f'{method:<20} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {gain:>+10.4f}')

        best_method = results[0][0]
        best_gain = results[0][4]
        print(f'\nBest Ensemble Method: {best_method} (Gain: {best_gain:+.4f})')


if __name__ == '__main__':
    run_experiment()
