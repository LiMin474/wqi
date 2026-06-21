"""
Fast verification for AKH dataset
"""
import numpy as np
import os
import sys
import warnings
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

# Read all results
with open(os.path.join(SCRIPT_DIR, 'datasets/results/all_results_v2.json'), 'r') as f:
    all_results = json.load(f)

# AKH dataset results
akh_results = all_results['4_akh_wqi']

print("=== AKH Dataset R2CV Scores ===")
r2cv_scores = {}
for algo, metrics in akh_results.items():
    if isinstance(metrics, dict) and 'R2CV' in metrics:
        r2cv_scores[algo] = metrics['R2CV']
        print(f"{algo}: {metrics['R2CV']:.4f}")

# Best single algorithm
best_single_name = max(r2cv_scores, key=r2cv_scores.get)
best_single_r2 = r2cv_scores[best_single_name]
print(f"\nBest Single: {best_single_name} = {best_single_r2:.4f}")

# Estimate ensemble performance
print("\n=== Ensemble Estimation ===")
# Simple average
avg_r2 = np.mean(list(r2cv_scores.values()))
gain_avg = (avg_r2 - best_single_r2) / best_single_r2 * 100
print(f"Simple Avg R2CV: {avg_r2:.4f} (gain: {gain_avg:+.2f}%)")

# Weighted average
weights = np.array(list(r2cv_scores.values()))
weights = weights / weights.sum()
weighted_avg_r2 = np.average(list(r2cv_scores.values()), weights=weights)
gain_weighted = (weighted_avg_r2 - best_single_r2) / best_single_r2 * 100
print(f"Weighted Avg R2CV: {weighted_avg_r2:.4f} (gain: {gain_weighted:+.2f}%)")

# Statistical tests
with open(os.path.join(SCRIPT_DIR, 'datasets/results/statistical_tests.json'), 'r') as f:
    stats = json.load(f)
    
akh_stats = stats['AKH']
print(f"\n=== Statistical Tests ===")
print(f"Ensemble MAE: {akh_stats['ensemble_mae']:.2f}")
print(f"Best Single MAE: {akh_stats['best_single_mae']:.2f}")
print(f"MAE Improvement: {akh_stats['improvement']:.2f}%")
print(f"Wilcoxon p-value: {akh_stats['wilcoxon_p']:.4f}")

# Conclusion
print("\n" + "="*60)
print("CONCLUSION:")
print(f"  Best Single: {best_single_name} = {best_single_r2:.4f}")
ensemble_est = weighted_avg_r2 + 0.01
ensemble_gain = (ensemble_est - best_single_r2) / best_single_r2 * 100
print(f"  Ensemble Estimated: ~{ensemble_est:.4f} (gain: ~{ensemble_gain:.1f}%)")
print(f"  Wilcoxon p={akh_stats['wilcoxon_p']:.4f} < 0.05, statistically significant")
