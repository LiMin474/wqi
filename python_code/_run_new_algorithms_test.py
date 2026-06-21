"""
测试新添加的三个算法：BOA、HHO-Lite、SSA-V2
在4个数据集上进行调优并记录结果
"""
import numpy as np
import os
import json
import sys

# 添加common_codes到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'common_codes'))

from a4_BOA_fitrnet_opt import a4_BOA_fitrnet_opt
from a4_HHO_Lite_fitrnet_opt import a4_HHO_Lite_fitrnet_opt
from a4_SSA_V2_fitrnet_opt import a4_SSA_V2_fitrnet_opt


def load_dataset(dataset_name):
    """加载指定数据集（.npz格式）"""
    data_dir = os.path.join(SCRIPT_DIR, 'datasets')
    
    datasets = {
        'Jajpur': '1_jajpur.npz',
        'WQI': '2_wqi_dataset.npz',
        'Sample': '3_sample_dataset.npz',
        'AKH': '4_akh_wqi.npz'
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = os.path.join(data_dir, datasets[dataset_name])
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"  Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def run_algorithm_on_dataset(algorithm_func, algorithm_name, X, y):
    """在数据集上运行指定算法"""
    try:
        print(f"\n  Running {algorithm_name}...")
        model, results = algorithm_func(X, y)
        print(f"  {algorithm_name} completed: R2={results['R2']:.4f}, R2CV={results['R2CV']:.4f}")
        return results
    except Exception as e:
        print(f"  Error running {algorithm_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数：在4个数据集上测试3个新算法"""
    algorithms = {
        'BOA': a4_BOA_fitrnet_opt,
        'HHO-Lite': a4_HHO_Lite_fitrnet_opt,
        'SSA-V2': a4_SSA_V2_fitrnet_opt
    }
    
    datasets = ['Jajpur', 'WQI', 'Sample', 'AKH']
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Testing on {dataset} dataset")
        print('='*60)
        
        X, y = load_dataset(dataset)
        all_results[dataset] = {}
        
        for algo_name, algo_func in algorithms.items():
            results = run_algorithm_on_dataset(algo_func, algo_name, X, y)
            if results is not None:
                all_results[dataset][algo_name] = {
                    'R2': float(results['R2']),
                    'R2CV': float(results['R2CV']),
                    'NumLayers': int(results['NumLayers']),
                    'Layer_1': int(results['Layer_1']),
                    'Layer_2': int(results['Layer_2']),
                    'Activation': results['Activation'],
                    'Alpha': float(results['Alpha'])
                }
    
    # 保存结果
    output_dir = os.path.join(SCRIPT_DIR, 'datasets', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'new_algorithms_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Results saved to:", output_file)
    print('='*60)
    
    # 打印汇总表格
    print("\n\n📊 新算法测试结果汇总：")
    print("-" * 80)
    print(f"{'数据集':<10} {'算法':<12} {'R2':<8} {'R2CV':<8} {'隐藏层':<10} {'激活函数':<10} {'Alpha':<12}")
    print("-" * 80)
    
    for dataset in datasets:
        for algo_name in algorithms.keys():
            if algo_name in all_results[dataset]:
                r = all_results[dataset][algo_name]
                layers = f"{r['NumLayers']}层({r['Layer_1']},{r['Layer_2']})"
                print(f"{dataset:<10} {algo_name:<12} {r['R2']:.4f}  {r['R2CV']:.4f}  {layers:<10} {r['Activation']:<10} {r['Alpha']:.2e}")
    
    print("\n📝 结果已保存到 new_algorithms_results.json")


if __name__ == '__main__':
    main()