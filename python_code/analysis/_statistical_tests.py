"""
统计显著性检验
对每个划分独立做检验，使用Fisher's方法合并p值，保证样本独立性
"""
import numpy as np
import json, os
from scipy.stats import wilcoxon, friedmanchisquare, ranksums, chi2


def fisher_method(p_values):
    """Fisher's方法合并多个独立检验的p值"""
    valid_p = [p for p in p_values if 0 < p <= 1]
    if len(valid_p) == 0:
        return 1.0, 0.0
    chi2_stat = -2 * np.sum(np.log(valid_p))
    df = 2 * len(valid_p)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return p_value, chi2_stat


def load_results(model_name='MLP-lbfgs'):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

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

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

    div_path = os.path.join(RESULTS_DIR, 'divergence_analysis.json')
    if not os.path.exists(div_path):
        print(f"请先运行 _analyze_divergence.py 生成 {div_path}")
        return
    with open(div_path, 'r') as f:
        divergence_data = json.load(f)

    ensemble_data = load_results(model_name)

    results = {}

    for ds_name in divergence_data.keys():
        print(f"\n=== {ds_name} 数据集统计检验 ===")

        has_splits = 'split_results' in divergence_data[ds_name]
        num_splits = divergence_data[ds_name].get('num_splits', 1)

        if has_splits:
            split_results = divergence_data[ds_name]['split_results']
            method_names = split_results[0]['method_names']
            n_methods = len(method_names)
            print(f"  使用 {num_splits} 次独立划分（样本不重复）")
        else:
            method_names = list(divergence_data[ds_name]['predictions'].keys())
            n_methods = len(method_names)
            print(f"  使用单次划分")

        ensemble_key = ds_name
        ens_result = ensemble_data.get(ensemble_key, {})
        single_results = ens_result.get('single_results', {})

        r2cv_values = []
        for method in method_names:
            r2cv = single_results.get(method, {}).get('R2CV', 0)
            r2cv_values.append(max(r2cv, 0))

        if sum(r2cv_values) > 0:
            weights = np.array(r2cv_values) / np.sum(r2cv_values)
        else:
            weights = np.ones(n_methods) / n_methods

        print(f"\n集成权重 (基于R²CV):")
        for m, w in zip(method_names, weights):
            print(f"  {m}: {w:.4f}")

        all_wilcoxon_p = []
        all_mannwhitney_p = []
        all_friedman_p = []
        all_improvements = []
        per_split_details = []

        if has_splits:
            print(f"\n逐划分检验:")
            for idx, split in enumerate(split_results):
                y_test = np.array(split['y_test'])
                preds = split['predictions']

                mae_values = {}
                for m in method_names:
                    mae_values[m] = np.mean(np.abs(y_test - np.array(preds[m])))

                best_single = min(mae_values, key=mae_values.get)
                best_single_mae = mae_values[best_single]

                ensemble_pred = np.zeros(len(y_test))
                for i, m in enumerate(method_names):
                    ensemble_pred += weights[i] * np.array(preds[m])
                ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))

                improvement = (best_single_mae - ensemble_mae) / best_single_mae * 100 if best_single_mae > 0 else 0
                all_improvements.append(improvement)

                ensemble_errors = np.abs(y_test - ensemble_pred)
                best_single_errors = np.abs(y_test - np.array(preds[best_single]))

                p_wilcoxon = 1.0
                try:
                    _, p_wilcoxon = wilcoxon(ensemble_errors, best_single_errors)
                except:
                    pass
                all_wilcoxon_p.append(p_wilcoxon)

                p_mannwhitney = 1.0
                try:
                    _, p_mannwhitney = ranksums(ensemble_errors, best_single_errors)
                except:
                    pass
                all_mannwhitney_p.append(p_mannwhitney)

                p_friedman = 1.0
                try:
                    error_matrix = []
                    for m in method_names:
                        error_matrix.append(np.abs(y_test - np.array(preds[m])))
                    error_matrix.append(ensemble_errors)
                    _, p_friedman = friedmanchisquare(*error_matrix)
                except:
                    pass
                all_friedman_p.append(p_friedman)

                per_split_details.append({
                    'split': idx + 1,
                    'n_samples': len(y_test),
                    'best_single': best_single,
                    'best_single_mae': best_single_mae,
                    'ensemble_mae': ensemble_mae,
                    'improvement_pp': improvement,
                    'wilcoxon_p': p_wilcoxon,
                    'mannwhitney_p': p_mannwhitney,
                    'friedman_p': p_friedman
                })

                print(f"  划分{idx+1}: n={len(y_test)}, best={best_single}, "
                      f"mae_single={best_single_mae:.4f}, mae_ensemble={ensemble_mae:.4f}, "
                      f"imp={improvement:+.2f}%, W={p_wilcoxon:.4f}, M={p_mannwhitney:.4f}, F={p_friedman:.4f}")

            fisher_wilcoxon_p, fisher_wilcoxon_chi2 = fisher_method(all_wilcoxon_p)
            fisher_mannwhitney_p, fisher_mannwhitney_chi2 = fisher_method(all_mannwhitney_p)
            fisher_friedman_p, fisher_friedman_chi2 = fisher_method(all_friedman_p)

            print(f"\nFisher's合并检验 ({num_splits}个独立划分):")
            print(f"  Wilcoxon: χ²={fisher_wilcoxon_chi2:.2f}, df={2*num_splits}, p={fisher_wilcoxon_p:.6f}")
            print(f"  Mann-Whitney: χ²={fisher_mannwhitney_chi2:.2f}, df={2*num_splits}, p={fisher_mannwhitney_p:.6f}")
            print(f"  Friedman: χ²={fisher_friedman_chi2:.2f}, df={2*num_splits}, p={fisher_friedman_p:.6f}")

            print(f"\n结论:")
            print(f"  集成 vs 最优单算法 (Wilcoxon): {'显著' if fisher_wilcoxon_p < 0.05 else '不显著'} (p<0.05)")
            print(f"  集成 vs 最优单算法 (Mann-Whitney): {'显著' if fisher_mannwhitney_p < 0.05 else '不显著'} (p<0.05)")
            print(f"  所有算法差异 (Friedman): {'显著差异' if fisher_friedman_p < 0.05 else '无显著差异'} (p<0.05)")

            avg_improvement = np.mean(all_improvements)
            std_improvement = np.std(all_improvements)

            results[ds_name] = {
                'num_splits': num_splits,
                'avg_improvement_pp': avg_improvement,
                'std_improvement_pp': std_improvement,
                'weights': {m: float(w) for m, w in zip(method_names, weights)},
                'fisher_wilcoxon_p': fisher_wilcoxon_p,
                'fisher_wilcoxon_chi2': fisher_wilcoxon_chi2,
                'fisher_mannwhitney_p': fisher_mannwhitney_p,
                'fisher_mannwhitney_chi2': fisher_mannwhitney_chi2,
                'fisher_friedman_p': fisher_friedman_p,
                'fisher_friedman_chi2': fisher_friedman_chi2,
                'significant_wilcoxon': fisher_wilcoxon_p < 0.05,
                'significant_mannwhitney': fisher_mannwhitney_p < 0.05,
                'significant_friedman': fisher_friedman_p < 0.05,
                'per_split_results': per_split_details
            }

        else:
            predictions = divergence_data[ds_name]['predictions']
            y_test = np.array(divergence_data[ds_name]['y_test'])

            mae_values = {}
            for m in method_names:
                mae_values[m] = np.mean(np.abs(y_test - np.array(predictions[m])))

            best_single = min(mae_values, key=mae_values.get)
            best_single_mae = mae_values[best_single]

            ensemble_pred = np.zeros(len(y_test))
            for i, m in enumerate(method_names):
                ensemble_pred += weights[i] * np.array(predictions[m])
            ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))

            improvement = (best_single_mae - ensemble_mae) / best_single_mae * 100 if best_single_mae > 0 else 0

            ensemble_errors = np.abs(y_test - ensemble_pred)
            best_single_errors = np.abs(y_test - np.array(predictions[best_single]))

            p_wilcoxon = 1.0
            try:
                _, p_wilcoxon = wilcoxon(ensemble_errors, best_single_errors)
            except:
                pass

            p_mannwhitney = 1.0
            try:
                _, p_mannwhitney = ranksums(ensemble_errors, best_single_errors)
            except:
                pass

            p_friedman = 1.0
            try:
                error_matrix = []
                for m in method_names:
                    error_matrix.append(np.abs(y_test - np.array(predictions[m])))
                error_matrix.append(ensemble_errors)
                _, p_friedman = friedmanchisquare(*error_matrix)
            except:
                pass

            print(f"\n单次划分检验 ({len(y_test)}个样本):")
            print(f"  Wilcoxon p值: {p_wilcoxon:.6f}")
            print(f"  Mann-Whitney p值: {p_mannwhitney:.6f}")
            print(f"  Friedman p值: {p_friedman:.6f}")

            print(f"\n结论:")
            print(f"  集成 vs 最优单算法 (Wilcoxon): {'显著' if p_wilcoxon < 0.05 else '不显著'} (p<0.05)")
            print(f"  集成 vs 最优单算法 (Mann-Whitney): {'显著' if p_mannwhitney < 0.05 else '不显著'} (p<0.05)")
            print(f"  所有算法差异 (Friedman): {'显著差异' if p_friedman < 0.05 else '无显著差异'} (p<0.05)")

            results[ds_name] = {
                'num_splits': 1,
                'avg_improvement_pp': improvement,
                'std_improvement_pp': 0,
                'weights': {m: float(w) for m, w in zip(method_names, weights)},
                'wilcoxon_p': p_wilcoxon,
                'mannwhitney_p': p_mannwhitney,
                'friedman_p': p_friedman,
                'significant_wilcoxon': p_wilcoxon < 0.05,
                'significant_mannwhitney': p_mannwhitney < 0.05,
                'significant_friedman': p_friedman < 0.05,
                'per_split_results': []
            }

    out_path = os.path.join(RESULTS_DIR, 'statistical_tests.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n统计检验结果已保存: {out_path}")

    csv_path = os.path.join(RESULTS_DIR, 'statistical_tests_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Dataset,NumSplits,AvgImprovementPP,StdImprovementPP,FisherWilcoxonP,FisherMannWhitneyP,FisherFriedmanP,Significant\n')
        for ds_name, r in results.items():
            sig = 'Yes' if r.get('significant_wilcoxon', False) or r.get('fisher_wilcoxon_p', 1) < 0.05 else 'No'
            f.write(f"{ds_name},{r['num_splits']},{r['avg_improvement_pp']:.2f},{r['std_improvement_pp']:.2f},"
                    f"{r.get('fisher_wilcoxon_p', r.get('wilcoxon_p', 1)):.6f},"
                    f"{r.get('fisher_mannwhitney_p', r.get('mannwhitney_p', 1)):.6f},"
                    f"{r.get('fisher_friedman_p', r.get('friedman_p', 1)):.6f},{sig}\n")

    print(f"统计检验汇总表已保存: {csv_path}")


if __name__ == '__main__':
    main()
