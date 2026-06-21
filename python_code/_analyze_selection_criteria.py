"""分析算法选型的数据依据"""
import json
import numpy as np

# Load data
with open('datasets/results/all_results_v2.json') as f:
    data = json.load(f)

# Also load GA/PSO/BOA results
with open('datasets/results/GA_PSO_BOA_results.json') as f:
    ga_pso_boa = json.load(f)

# Merge
for ds in ga_pso_boa:
    for algo in ga_pso_boa[ds]:
        data[ds][algo] = ga_pso_boa[ds][algo]

# Focus on main algorithms
main_algos = ['Bayesian', 'DE', 'SHADE', 'APSM-jSO (2023)', 'CMA-ES', 'GA', 'PSO', 'BOA']
datasets = ['1_jajpur', '2_wqi_dataset', '3_sample_dataset', '4_akh_wqi']

# Extract R2CV for each algorithm on each dataset
r2cv_matrix = {}
for algo in main_algos:
    r2cv_matrix[algo] = []
    for ds in datasets:
        if algo in data[ds]:
            r2cv_matrix[algo].append(data[ds][algo]['R2CV'])
        else:
            r2cv_matrix[algo].append(None)

# Print R2CV table
print('='*80)
print('R2CV Matrix (8 algorithms x 4 datasets)')
print('='*80)
header = 'Algorithm          ' + '  Jajpur' + '  Indian' + '  Ireland' + '    AKH' + '    Mean'
print(header)
print('-'*80)
for algo in main_algos:
    vals = r2cv_matrix[algo]
    mean_val = np.mean([v for v in vals if v is not None])
    row = f'{algo:<18}'
    for v in vals:
        if v is not None:
            row += f'{v:>10.4f}'
        else:
            row += '     N/A'
    row += f'{mean_val:>10.4f}'
    print(row)

# Calculate ranking for each dataset
print()
print('='*80)
print('Ranking per dataset (1=best, 8=worst)')
print('='*80)
ranks = {}
for algo in main_algos:
    ranks[algo] = []

for i, ds in enumerate(datasets):
    vals = [(algo, r2cv_matrix[algo][i]) for algo in main_algos if r2cv_matrix[algo][i] is not None]
    vals_sorted = sorted(vals, key=lambda x: -x[1])  # descending by R2CV
    for rank, (algo, _) in enumerate(vals_sorted, 1):
        ranks[algo].append(rank)

header = 'Algorithm          ' + '  Jajpur' + '  Indian' + '  Ireland' + '    AKH' + ' AvgRank'
print(header)
print('-'*80)
avg_ranks = {}
for algo in main_algos:
    avg_rank = np.mean(ranks[algo])
    avg_ranks[algo] = avg_rank
    row = f'{algo:<18}'
    for r in ranks[algo]:
        row += f'{r:>10}'
    row += f'{avg_rank:>10.2f}'
    print(row)

# Sort by average rank
print()
print('Sorted by Average Rank:')
for algo in sorted(main_algos, key=lambda x: avg_ranks[x]):
    print(f'  {algo:<18}: AvgRank = {avg_ranks[algo]:.2f}')

# Calculate disagreement: std of R2CV across datasets
print()
print('='*80)
print('Disagreement (std of R2CV across datasets)')
print('Higher std = algorithm behaves differently on different datasets = more diverse')
print('='*80)
disagreement = {}
for algo in main_algos:
    vals = [v for v in r2cv_matrix[algo] if v is not None]
    disagreement[algo] = np.std(vals)

for algo in sorted(main_algos, key=lambda x: -disagreement[x]):
    print(f'{algo:<18}: std = {disagreement[algo]:.4f}')

# Summary: which algorithms to select?
print()
print('='*80)
print('SELECTION CRITERIA SUMMARY')
print('='*80)
print()
print('Criterion 1: Average Rank (lower is better)')
top5_rank = sorted(main_algos, key=lambda x: avg_ranks[x])[:5]
print(f'  Top 5 by AvgRank: {top5_rank}')
print()
print('Criterion 2: Disagreement (higher = more diverse)')
top5_disagree = sorted(main_algos, key=lambda x: -disagreement[x])[:5]
print(f'  Top 5 by Disagreement: {top5_disagree}')
print()
print('Criterion 3: At least one dataset optimal')
optimal_algos = set()
for i, ds in enumerate(datasets):
    vals = [(algo, r2cv_matrix[algo][i]) for algo in main_algos if r2cv_matrix[algo][i] is not None]
    best_algo = max(vals, key=lambda x: x[1])[0]
    optimal_algos.add(best_algo)
    print(f'  {ds}: optimal = {best_algo} (R2CV={r2cv_matrix[best_algo][i]:.4f})')
print(f'  Algorithms that won at least once: {sorted(optimal_algos)}')
print()
print('='*80)
print('RECOMMENDATION')
print('='*80)
print()
print('Based on data:')
print(f'  - Top 5 by AvgRank: {top5_rank}')
print(f'  - Algorithms that won at least once: {sorted(optimal_algos)}')
print()
# Intersection
intersection = set(top5_rank) & optimal_algos
print(f'  Intersection (good rank + won at least once): {sorted(intersection)}')
print()
# BOA status
boa_rank = avg_ranks['BOA']
boa_won = 'BOA' in optimal_algos
print(f'  BOA status: AvgRank={boa_rank:.2f}, Won={boa_won}')
print()
print('Note: BOA (Baboon, 2026) never won, but is the newest algorithm.')
print('      If teacher wants new algorithms, consider including BOA.')