"""
模型配置工厂 — 统一接口供 EA 优化器调用

每个模型配置是一个 dict:
{
    'name': str,               # 模型名称
    'n_params': int,           # 参数维度
    'param_names': list[str],  # 参数名列表
    'decode': callable,        # decode(x) -> params_tuple
    'evaluate': callable,      # evaluate(params, X, y, cv_splits) -> (target, output_dict)
    'get_param_dict': callable,# get_param_dict(params) -> {name: value}
}

output_dict 必须包含: {'R2': float, 'R2CV': float, 'Mdl': sklearn estimator}
target = 1 - R2CV (最小化)
"""
from .mlp_lbfgs import MLP_LBFGS_CONFIG
from .cnn_1d import CNN_1D_CONFIG
from .xgboost_model import XGBOOST_CONFIG

try:
    from .cnn_1d_pytorch import CNN_1D_PT_CONFIG
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

MODEL_REGISTRY = {
    'MLP-lbfgs': MLP_LBFGS_CONFIG,
    '1D-CNN': CNN_1D_CONFIG,
    'XGBoost': XGBOOST_CONFIG,
}

if _PT_AVAILABLE:
    MODEL_REGISTRY['1D-CNN-PT'] = CNN_1D_PT_CONFIG

DEFAULT_CONFIG = MLP_LBFGS_CONFIG  # 向后兼容


def get_model_config(name: str) -> dict:
    """根据名称获取模型配置"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]