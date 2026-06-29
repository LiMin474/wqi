"""深度验证AKH数据正确性 - 对比三数据集一致性"""
import numpy as np
import json

with open('python_code/results/unified_ensemble_results.json') as f:
    all_data = json.load(f)

print('=' * 70)
print('三数据集一致性验证')
print('=' * 70)

for ds_name in ['Jajpur', 'Irish', 'AKH']:
    d = all_data[ds_name]
    single = d['single_results']
    ea = ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']
    r2cvs = [single[a]['R2CV'] for a in ea]
    ensemble_r2cv = d['ensemble_results']['WeightedAvg']['R2CV']
    best_single = max(r2cvs)
    improvement = (ensemble_r2cv - best_single) / best_single * 100

    print(f'\n{ds_name} (n={np.load(f"python_code/datasets/{ds_name.lower()}_..." if ds_name!="Jajpur" else "...") })')

# 直接打印
datasets_info = {
    'Jajpur': {'file': '1_jajpur_groundwater.npz', 'label': 'Jajpur-Groundwater'},
    'Irish': {'file': '2_irish_river.npz', 'label': 'Irish-River-CCME'},
    'AKH': {'file': '3_akh_wqi.npz', 'label': 'AKH-WQI Surface Water'},
}

print(f"{'数据集':<20} {'样本':>6} {'特征':>4} {'单算法R²CV范围':>20} {'集成R²CV':>10} {'提升%':>8}")
print('-' * 70)

for key, info in datasets_info.items():
    d = np.load(f'python_code/datasets/{info["file"]}', allow_pickle=True)
    n_samples = d['X'].shape[0]
    n_features = d['X'].shape[1]
    y_mean = d['y'].mean()
    y_std = d['y'].std()

    jd = all_data[key]
    single = jd['single_results']
    ea_list = ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']
    r2cvs = [single[a]['R2CV'] for a in ea_list]
    ensemble = jd['ensemble_results']['WeightedAvg']['R2CV']
    best_single = max(r2cvs)
    imp = (ensemble - best_single) / best_single * 100

    print(f"{info['label']:<20} {n_samples:>6} {n_features:>4} {min(r2cvs):.4f}~{max(r2cvs):.4f} {ensemble:>10.4f} {imp:>+7.2f}%")
    print(f"{'':20} {'':6} {'':4}  y: {y_mean:.1f}+-{y_std:.1f}")

print('\n' + '=' * 70)
print('AKH各算法参数对比')
print('=' * 70)

print(f"{'算法':<12} {'R2CV':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Time':>8}")
print('-' * 55)
akh = all_data['AKH']['single_results']
for a in ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']:
    r = akh[a]
    print(f"{a:<12} {r['R2CV']:>8.4f} {r['R2']:>8.4f} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['Time']:>8.1f}")

wa = all_data['AKH']['ensemble_results']['WeightedAvg']
print(f"{'WeightedAvg':<12} {wa['R2CV']:>8.4f} {wa['R2']:>8.4f} {wa['RMSE']:>8.3f} {wa['MAE']:>8.3f} {'N/A':>8}")

# 检查集成是否真的比所有单算法好
print('\n' + '=' * 70)
print('集成 vs 最佳单算法 逐样本误差对比')
print('=' * 70)
import pandas as pd
df = pd.read_csv('python_code/results/scatter_AKH.csv')

for algo in ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']:
    algo_mae = abs(df['Actual'] - df[algo]).mean()
    algo_rmse = np.sqrt((abs(df['Actual'] - df[algo])**2).mean())
    print(f'{algo:<12} MAE={algo_mae:.3f}  RMSE={algo_rmse:.3f}')

ens_mae = abs(df['Actual'] - df['WeightedAvg_Pred']).mean()
ens_rmse = np.sqrt((abs(df['Actual'] - df['WeightedAvg_Pred'])**2).mean())
print(f'{'Ensemble':<12} MAE={ens_mae:.3f}  RMSE={ens_rmse:.3f}')

# 统计集成比单算法好的样本比例
best_algo_pred = df[['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']].min(axis=1)
better_count = sum(abs(df['Actual'] - df['WeightedAvg_Pred']) < abs(df['Actual'] - best_algo_pred))
print(f'\n集成优于所有单算法的样本: {better_count}/{len(df)} ({better_count/len(df)*100:.1f}%)')

print('\n✅ 验证完成')
