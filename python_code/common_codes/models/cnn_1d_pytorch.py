"""
1D-CNN (PyTorch版本) — 一维卷积神经网络
参数空间: 卷积核数、核大小、学习率、batch_size、dropout
使用 PyTorch + skorch 保持 sklearn 兼容接口，支持 GPU 加速
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Reshape1D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.expand_dims(X, axis=-1)


_CNN_PT_IMPORTED = False


def _ensure_imports():
    global _CNN_PT_IMPORTED
    if not _CNN_PT_IMPORTED:
        global torch, nn, NeuralNetRegressor, StandardScaler, Pipeline, cross_val_score, CNN1DModel
        import torch
        import torch.nn as nn
        from skorch import NeuralNetRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        class CNN1DModel(nn.Module):
            def __init__(self, n_features, filters, kernel_size, dropout_rate):
                super().__init__()
                self.conv1 = nn.Conv1d(1, filters, kernel_size, padding='same')
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout_rate)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(filters, 1)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        torch.manual_seed(1)
        _CNN_PT_IMPORTED = True


def cnn_1d_pt_decode(x):
    filters_idx = int(x[0] * 3)
    filters = [16, 32, 64][min(filters_idx, 2)]
    kernel_size = int(round(2 + x[1] * 2))
    kernel_size = max(2, min(4, kernel_size))
    lr = 10.0 ** (-4.0 + x[2] * 3.0)
    batch_size_idx = int(x[3] * 4)
    batch_size = [8, 16, 32, 64][min(batch_size_idx, 3)]
    dropout_rate = x[4] * 0.5
    return filters, kernel_size, lr, batch_size, dropout_rate


def cnn_1d_pt_evaluate(params, XX, YY, cvss):
    _ensure_imports()
    filters, kernel_size, lr, batch_size, dropout_rate = params

    n_features = XX.shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    YY_pt = YY.astype(np.float32).reshape(-1, 1)

    Mdl = Pipeline([
        ('scaler', StandardScaler()),
        ('reshape', Reshape1D()),
        ('cnn', NeuralNetRegressor(
            CNN1DModel(n_features=n_features, filters=filters,
                       kernel_size=kernel_size, dropout_rate=dropout_rate),
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=lr,
            batch_size=batch_size,
            max_epochs=30,
            verbose=0,
            device=device,
        ))
    ])

    Mdl.fit(XX, YY_pt)

    SST = np.sum((YY - np.mean(YY)) ** 2)
    y_pred = Mdl.predict(XX).ravel()
    SSEmdl = np.sum((YY - y_pred) ** 2)

    cv_scores = cross_val_score(Mdl, XX, YY_pt, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -cv_scores.sum() * len(YY) / len(cv_scores)
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def cnn_1d_pt_get_param_dict(params):
    filters, kernel_size, lr, batch_size, dropout_rate = params
    return {
        'Filters': filters,
        'KernelSize': kernel_size,
        'LearningRate': f'{lr:.2e}',
        'BatchSize': batch_size,
        'Dropout': f'{dropout_rate:.3f}',
    }


CNN_1D_PT_CONFIG = {
    'name': '1D-CNN-PT',
    'n_params': 5,
    'param_names': ['Filters', 'KernelSize', 'LearningRate', 'BatchSize', 'Dropout'],
    'decode': cnn_1d_pt_decode,
    'evaluate': cnn_1d_pt_evaluate,
    'get_param_dict': cnn_1d_pt_get_param_dict,
}
