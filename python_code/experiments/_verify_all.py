"""深度验证三数据集一致性"""
import numpy as np
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')


def load_results(model_name='MLP-lbfgs'):
    path = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'unified_ensemble_results.json')
        print(f"[WARNING] comprehensive_results.json not found, falling back to {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    if model_name in data:
        return data[model_name]
    return data


def main():
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'MLP-lbfgs'
    all_data = load_results(model_name)

    print('=' * 70)
    print(f'三数据集一致性验证 (模型: {model_name})')
    print('=' * 70)

    datasets_info = {
        'Jajpur': {'file': '1_jajpur.npz', 'label': 'Jajpur-Groundwater'},
        'Irish': {'file': '2_irish_river.npz', 'label': 'Irish-River-CCME'},
        'AKH': {'file': '3_akh_wqi.npz', 'label': 'AKH-WQI Surface Water'},
    }

    print(f"{'数据集':<20} {'样本':>6} {'特征':>4} {'单算法R²CV范围':>20} {'集成R²CV':>10} {'提升%':>8}")
    print('-' * 70)

    for key, info in datasets_info.items():
        d = np.load(os.path.join(DATASET_DIR, info['file']), allow_pickle=True)
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
    print(f'{model_name} - AKH各算法参数对比')
    print('=' * 70)

    print(f"{'算法':<12} {'R2CV':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Time':>8}")
    print('-' * 55)
    akh = all_data['AKH']['single_results']
    for a in ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']:
        r = akh[a]
        print(f"{a:<12} {r['R2CV']:>8.4f} {r['R2']:>8.4f} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['Time']:>8.1f}")

    wa = all_data['AKH']['ensemble_results']['WeightedAvg']
    print(f"{'WeightedAvg':<12} {wa['R2CV']:>8.4f} {wa['R2']:>8.4f} {wa['RMSE']:>8.3f} {wa['MAE']:>8.3f} {'N/A':>8}")

    print('\n' + '=' * 70)
    print(f'{model_name} - 集成 vs 最佳单算法 逐样本误差对比 (AKH)')
    print('=' * 70)
    import pandas as pd
    csv_path = os.path.join(RESULTS_DIR, f'scatter_{model_name}_AKH.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(RESULTS_DIR, 'scatter_AKH.csv')
        print(f"[WARNING] Using old format: {csv_path}")
    df = pd.read_csv(csv_path)

    for algo in ['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']:
        algo_mae = abs(df['Actual'] - df[algo]).mean()
        algo_rmse = np.sqrt((abs(df['Actual'] - df[algo])**2).mean())
        print(f'{algo:<12} MAE={algo_mae:.3f}  RMSE={algo_rmse:.3f}')

    ens_mae = abs(df['Actual'] - df['WeightedAvg_Pred']).mean()
    ens_rmse = np.sqrt((abs(df['Actual'] - df['WeightedAvg_Pred'])**2).mean())
    print(f'{'Ensemble':<12} MAE={ens_mae:.3f}  RMSE={ens_rmse:.3f}')

    best_algo_pred = df[['DE','SHADE','CMA-ES','NRBO','BOA','HHO-Lite']].min(axis=1)
    better_count = sum(abs(df['Actual'] - df['WeightedAvg_Pred']) < abs(df['Actual'] - best_algo_pred))
    print(f'\n集成优于所有单算法的样本: {better_count}/{len(df)} ({better_count/len(df)*100:.1f}%)')

    print('\n✅ 验证完成')


if __name__ == '__main__':
    main()
