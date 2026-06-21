"""
快速验证：只跑AKH数据集的5算法集成
"""
import numpy as np
import os
import sys
import warnings
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_SHADE_fitrnet_opt import a4_SHADE_fitrnet_opt
from common_codes.a4_APSM_jSO_fitrnet_opt import a4_APSM_jSO_fitrnet_opt
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_NRBO_fitrnet_opt import a4_NRBO_fitrnet_opt


def load_dataset(name):
    data_path = os.path.join(SCRIPT_DIR, 'datasets', f'{name}.npz')
    data = np.load(data_path, allow_pickle=True)
    return np.asarray(data['X'], dtype=float), np.asarray(data['y'], dtype=float).ravel()


print("Loading AKH dataset...")
X, y = load_dataset('4_akh_wqi')
print(f"Dataset size: n={len(y)}")

# 划分训练/测试
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# 5-fold CV for OOF predictions
methods = {
    'DE': a4_DE_fitrnet_opt,
    'SHADE': a4_SHADE_fitrnet_opt,
    'APSM-jSO': a4_APSM_jSO_fitrnet_opt,
    'CMA-ES': a4_CMAES_fitrnet_opt,
    'NRBO': a4_NRBO_fitrnet_opt,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_train = {}
test_preds = {}
r2cv_scores = {}

for name, func in methods.items():
    print(f"\nTraining {name}...")
    
    # OOF predictions on training set
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx in kf.split(X_train):
        Mdl, _ = func(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = Mdl.predict(X_train[va_idx])
    
    oof_train[name] = oof
    r2cv_scores[name] = r2_score(y_train, oof)
    print(f"  {name} R2CV: {r2cv_scores[name]:.4f}")
    
    # Final model on full train set for test prediction
    Mdl_full, _ = func(X_train, y_train)
    test_preds[name] = Mdl_full.predict(X_test)
    print(f"  {name} Test R2: {r2_score(y_test, test_preds[name]):.4f}")

# Best single algorithm
best_single = max(r2cv_scores.items(), key=lambda x: x[1])
print(f"\n=== Best Single Algorithm: {best_single[0]} (R2CV={best_single[1]:.4f}) ===")

# LR Stacking
print("\n=== LR Stacking ===")
X_train_stack = np.column_stack([oof_train[name] for name in methods.keys()])
X_test_stack = np.column_stack([test_preds[name] for name in methods.keys()])

lr = LinearRegression()
lr.fit(X_train_stack, y_train)
lr_test_pred = lr.predict(X_test_stack)
lr_r2 = r2_score(y_test, lr_test_pred)

print(f"LR Stacking Test R2: {lr_r2:.4f}")
print(f"LR Stacking Gain: {(lr_r2 - best_single[1]):.4f} ({(lr_r2 - best_single[1])/best_single[1]*100:.2f}%)")

# Simple Average
print("\n=== Simple Average ===")
sa_pred = np.mean(X_test_stack, axis=1)
sa_r2 = r2_score(y_test, sa_pred)
print(f"SimpleAvg Test R2: {sa_r2:.4f}")

# Weighted Average
print("\n=== Weighted Average ===")
weights = np.array(list(r2cv_scores.values()))
weights = weights / weights.sum()
wa_pred = np.average(X_test_stack, axis=1, weights=weights)
wa_r2 = r2_score(y_test, wa_pred)
print(f"WeightedAvg Test R2: {wa_r2:.4f}")

# Summary
print("\n" + "="*60)
print("FINAL RESULT (AKH Dataset)")
print("="*60)
print(f"Best Single: {best_single[0]} = {best_single[1]:.4f}")
print(f"SimpleAvg:   {sa_r2:.4f} (gain: {(sa_r2-best_single[1])/best_single[1]*100:+.2f}%)")
print(f"WeightedAvg: {wa_r2:.4f} (gain: {(wa_r2-best_single[1])/best_single[1]*100:+.2f}%)")
print(f"LRStacking: {lr_r2:.4f} (gain: {(lr_r2-best_single[1])/best_single[1]*100:+.2f}%)")
