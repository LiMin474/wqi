"""Compare DE-FS results with the paper's feature selection methods:
   - PPI (Predictor Profile Importance) from PPE
   - MRMR (Minimum Redundancy Maximum Relevance)
   - SBE (Sequential Backward Elimination)
   - Bayesian Feature Selection (BayesFS)
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'common_codes'))
sys.path.insert(0, os.path.join('.', 'variable_selection'))

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from data_loader import load_wqdata, load_stdwt
from common_codes.a2_GWQI import a2_GWQI
from common_codes.a4_DE_feature_selection import a4_DE_feature_selection, VAR_NAMES, SumSqr_DE_FS, decode_params
from common_codes.a4_DE_feature_selection import a4_DE_feature_selection

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, 'saved_models')
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
wqdata = load_wqdata()
stdwt = load_stdwt()
for col in ['TH']:
    if col in wqdata.columns: wqdata = wqdata.drop(columns=[col])
    if col in stdwt.columns: stdwt = stdwt.drop(columns=[col])
BISd = stdwt.values.astype(float)
X = wqdata.iloc[:, 3:15].values.astype(float)
GQ = a2_GWQI(X.copy(), BISd)
print(f'Data: {X.shape[0]} samples, {X.shape[1]} features')
print(f'Features: {VAR_NAMES}\n')

# ============================================================
# 2. PPI ranking from PPE_mean.npz
# ============================================================
print('='*60)
print('METHOD 1: PPI (Predictor Profile Importance)')
print('='*60)
ppe_data = np.load(os.path.join(save_dir, 'PPE_mean.npz'), allow_pickle=True)
meanloss = ppe_data['meanloss']
# meanloss structure: [mean, ann_imp, rf_imp] or similar
# From MATLAB: meanloss{2} = ANN importance, meanloss{3} = RF importance
ann_imp = meanloss[1]  # ANN importance (higher = more important)
rf_imp = meanloss[2]   # RF importance

ann_rank = np.argsort(ann_imp)[::-1]
rf_rank = np.argsort(rf_imp)[::-1]

print(f'ANN importance ranking:')
for i, idx in enumerate(ann_rank):
    print(f'  {i+1}. {VAR_NAMES[idx]} (imp={ann_imp[idx]:.4f})')

print(f'\nRF importance ranking:')
for i, idx in enumerate(rf_rank):
    print(f'  {i+1}. {VAR_NAMES[idx]} (imp={rf_imp[idx]:.4f})')

# ============================================================
# 3. MRMR ranking
# ============================================================
print('\n' + '='*60)
print('METHOD 2: MRMR (Minimum Redundancy Maximum Relevance)')
print('='*60)
mrmr_scores = mutual_info_regression(X, GQ)
mrmr_rank = np.argsort(mrmr_scores)[::-1]
print(f'MRMR ranking (mutual information score):')
for i, idx in enumerate(mrmr_rank):
    print(f'  {i+1}. {VAR_NAMES[idx]} (MI={mrmr_scores[idx]:.4f})')

# ============================================================
# 4. SBE (Sequential Backward Elimination) - simplified version
# ============================================================
print('\n' + '='*60)
print('METHOD 3: SBE (Sequential Backward Elimination)')
print('='*60)
# Start with all features, iteratively remove the one whose removal hurts least
# Use the SumSqr_DE_FS objective (1-R2CV) as the criterion
from sklearn.model_selection import KFold
from common_codes.a4_DE_feature_selection import SumSqr_DE_FS, decode_params

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cvss = list(kf.split(X))

# Default MLP params for SBE
# Use a reasonable fixed architecture
x_default = np.ones(17)
x_default[12] = 2/3  # relu
x_default[13] = 0.2  # 1 layer
x_default[14] = 0.5  # 6 neurons
default_params = decode_params(x_default)

selected = list(range(12))
sbe_history = []
r2cv_all = 0  # Previous R2CV

print(f'Starting with all 12 features...')
target_full, output_full = SumSqr_DE_FS(default_params, X, GQ, cvss, max_iter=500)
print(f'  All features: R2CV={output_full["R2CV"]:.4f}')
r2cv_all = output_full['R2CV']
sbe_history.append({'selected': selected.copy(), 'R2CV': r2cv_all})

while len(selected) > 2:
    best_removal = None
    best_r2cv = -np.inf

    for i in range(len(selected)):
        candidate = selected[:i] + selected[i+1:]
        mask = np.zeros(12, dtype=bool)
        mask[candidate] = True
        params = (mask, default_params[1], default_params[2],
                  default_params[3], default_params[4], default_params[5])
        target, output = SumSqr_DE_FS(params, X, GQ, cvss, max_iter=500)
        if output['R2CV'] > best_r2cv:
            best_r2cv = output['R2CV']
            best_removal = i

    removed_feat = selected.pop(best_removal)
    print(f'  Remove {VAR_NAMES[removed_feat]:4s}: R2CV={best_r2cv:.4f} '
          f'(remaining: {len(selected)})')

    # Stop if R2CV drops below 90% of original
    if best_r2cv < 0.90 * r2cv_all:
        selected.insert(best_removal, removed_feat)
        print(f'  >>> STOP: R2CV dropped below 90% threshold')
        break

    sbe_history.append({'selected': selected.copy(), 'R2CV': best_r2cv})
    r2cv_all = best_r2cv

print(f'\nSBE final selected ({len(selected)}): {[VAR_NAMES[i] for i in selected]}')

# ============================================================
# 5. Run DE-FS 
# ============================================================
print('\n' + '='*60)
print('METHOD 4: DE-FS (Our method)')
print('='*60)
Modells_ann_fs, Opttable_ann_fs = a4_DE_feature_selection(X, GQ)

# ============================================================
# 6. Comprehensive Comparison Table
# ============================================================
print('\n\n' + '='*80)
print('            COMPREHENSIVE FEATURE SELECTION COMPARISON')
print('='*80)

# Collect which features each method selects
def feat_set(indices):
    """Return set of feature indices"""
    return set(indices)

def feat_names(indices):
    return [VAR_NAMES[i] for i in indices]

# PPI-ANN top 8 (select same number as DE-FS)
ppi_ann_top8 = set(ann_rank[:8])
# PPI-RF top 8
ppi_rf_top8 = set(rf_rank[:8])
# MRMR top 8
mrmr_top8 = set(mrmr_rank[:8])
# SBE selected
sbe_set = set(selected)
# DE-FS selected
de_fs_set = set(np.where(Opttable_ann_fs['FeatureMask'])[0])

methods = {
    'PPI-ANN (top 8)': ppi_ann_top8,
    'PPI-RF (top 8)': ppi_rf_top8,
    'MRMR (top 8)': mrmr_top8,
    'SBE': sbe_set,
    'DE-FS': de_fs_set,
}

# Print feature selection matrix
print(f'\n{"Feature":<8}', end='')
for method in methods:
    print(f'{method:<18}', end='')
print()

for j in range(12):
    print(f'{VAR_NAMES[j]:<8}', end='')
    for method in methods:
        mark = '+' if j in methods[method] else '-'
        print(f'{mark:<18}', end='')
    print()

# Count agreement
print(f'\n{"Agreement with DE-FS":<30}', end='')
for method in methods:
    if method == 'DE-FS':
        print(f'{"-":<18}', end='')
    else:
        common = len(methods[method] & de_fs_set)
        print(f'{common:<18}', end='')
print()

# How many features each selects
print(f'\n{"N_Features":<30}', end='')
for method in methods:
    print(f'{len(methods[method]):<18}', end='')
print()

# Which specific features each selects
print(f'\n{"Selected Features":<30} {"Count":<8} {"Features"}')
for method, feat_set_obj in methods.items():
    names = [VAR_NAMES[i] for i in sorted(feat_set_obj)]
    print(f'{method:<30} {len(names):<8} {names}')

# Which features are dropped
print(f'\n{"Dropped Features":<30} {"Count":<8} {"Features"}')
all_feats = set(range(12))
for method, feat_set_obj in methods.items():
    dropped = sorted(all_feats - feat_set_obj)
    names = [VAR_NAMES[i] for i in dropped]
    print(f'{method:<30} {len(names):<8} {names}')

print('\n' + '='*80)

# Summary analysis
print('\n=== KEY INSIGHTS ===')
print(f'DE-FS dropped: {Opttable_ann_fs["DroppedVars"]}')

# Features never dropped by any method
always_kept = all_feats
for m in methods.values():
    always_kept = always_kept & m
always_kept_names = [VAR_NAMES[i] for i in sorted(always_kept)]
print(f'Features kept by ALL methods: {always_kept_names}')

# Features dropped by most methods
dropped_count = {j: 0 for j in range(12)}
for m in methods.values():
    for j in range(12):
        if j not in m:
            dropped_count[j] += 1
freq_dropped = sorted([(j, c) for j, c in dropped_count.items() if c >= 2], key=lambda x: -x[1])
print(f'Features dropped by ≥2 methods:')
for j, c in freq_dropped:
    print(f'  {VAR_NAMES[j]}: {c}/4 methods')

# Save for paper
np.savez(os.path.join(save_dir, 'feature_selection_comparison.npz'),
         var_names=VAR_NAMES,
         ppi_ann_rank=ann_rank, ppi_rf_rank=rf_rank,
         mrmr_rank=mrmr_rank,
         sbe_selected=np.array(selected),
         de_fs_mask=Opttable_ann_fs['FeatureMask'],
         ppi_ann_imp=ann_imp, ppi_rf_imp=rf_imp,
         mrmr_scores=mrmr_scores)
print(f'\nComparison data saved to: {os.path.join(save_dir, "feature_selection_comparison.npz")}')
print('Done!')