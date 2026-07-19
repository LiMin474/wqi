"""
1D-CNN — 一维卷积神经网络
参数空间: 卷积核数、核大小、学习率、batch_size、dropout
使用 Keras + scikeras 保持 sklearn 兼容接口
"""
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

# 抑制 TensorFlow 烦人警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Reshape1D(BaseEstimator, TransformerMixin):
    """将 2D 特征矩阵转换为 3D (n_samples, n_features, 1)，适配 Conv1D 输入"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.expand_dims(X, axis=-1)


_CNN_IMPORTED = False


def _ensure_imports():
    global _CNN_IMPORTED
    if not _CNN_IMPORTED:
        global Sequential, Conv1D, GlobalAveragePooling1D, Dense, Dropout, Adam
        global KerasRegressor, StandardScaler, Pipeline, cross_val_score
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import tensorflow as tf
            tf.random.set_seed(1)
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from scikeras.wrappers import KerasRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import Pipeline
        _CNN_IMPORTED = True


def cnn_1d_decode(x):
    """Decode [0,1]^5 vector into 1D-CNN hyperparameters."""
    filters_idx = int(x[0] * 3)
    filters = [16, 32, 64][min(filters_idx, 2)]
    kernel_size = int(round(2 + x[1] * 2))
    kernel_size = max(2, min(4, kernel_size))
    lr = 10.0 ** (-4.0 + x[2] * 3.0)
    batch_size_idx = int(x[3] * 4)
    batch_size = [8, 16, 32, 64][min(batch_size_idx, 3)]
    dropout_rate = x[4] * 0.5
    return filters, kernel_size, lr, batch_size, dropout_rate


def cnn_1d_evaluate(params, XX, YY, cvss):
    """训练 1D-CNN，返回 (target, output_dict)"""
    _ensure_imports()
    filters, kernel_size, lr, batch_size, dropout_rate = params

    n_features = XX.shape[1]

    def build_fn():
        model = Sequential([
            Conv1D(filters, kernel_size, activation='relu',
                   input_shape=(n_features, 1), padding='same'),
            Dropout(dropout_rate),
            GlobalAveragePooling1D(),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    Mdl = Pipeline([
        ('scaler', StandardScaler()),
        ('reshape', Reshape1D()),
        ('cnn', KerasRegressor(
            model=build_fn,
            batch_size=batch_size,
            epochs=10,
            verbose=0,
            random_state=1,
        ))
    ])

    Mdl.fit(XX, YY)

    SST = np.sum((YY - np.mean(YY)) ** 2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred) ** 2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -cv_scores.sum() * len(YY) / len(cv_scores)
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def cnn_1d_get_param_dict(params):
    """将参数元组转为命名字典"""
    filters, kernel_size, lr, batch_size, dropout_rate = params
    return {
        'Filters': filters,
        'KernelSize': kernel_size,
        'LearningRate': f'{lr:.2e}',
        'BatchSize': batch_size,
        'Dropout': f'{dropout_rate:.3f}',
    }


CNN_1D_CONFIG = {
    'name': '1D-CNN',
    'n_params': 5,
    'param_names': ['Filters', 'KernelSize', 'LearningRate', 'BatchSize', 'Dropout'],
    'decode': cnn_1d_decode,
    'evaluate': cnn_1d_evaluate,
    'get_param_dict': cnn_1d_get_param_dict,
}