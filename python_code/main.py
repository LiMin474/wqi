"""
Main analysis script - Python equivalent of WQA_analysis_Jajpur.m
Water Quality Assessment (WQA) analysis for Jajpur district.

Usage:
    python main.py

Requires data files:
    - a0_Postmonsoon_JAJAPUR.mat (or CSV exports)
    - b0_X_GQ.mat (or CSV exports)
    Or run export_data_to_csv.m in MATLAB first.
"""
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, os.path.join(_BASE_DIR, 'common_codes'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'Bootstrap_new'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'Map_Codes'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'variable_selection'))

# Import data loader
from data_loader import load_GQ, load_X0, load_wqdata, load_stdwt, load_BISd, load_all_data, save_model_results

# Import common_codes modules
from common_codes.a0_statistics import a0_statistics_X
from common_codes.a1_heatmap_boxplot import a1_heatmap_boxplot
from common_codes.a2_GWQI import a2_GWQI
from common_codes.a3_Bayesopt_rf_model import a3_Bayesopt_rf_model
from common_codes.a4_Bayesian_fitrnet_opt import a4_Bayesian_fitrnet_opt
from common_codes.a4_DE_fitrnet_opt import a4_DE_fitrnet_opt
from common_codes.a4_DE_fitrnet_opt_reg import a4_DE_fitrnet_opt_reg
from common_codes.a4_CMAES_fitrnet_opt import a4_CMAES_fitrnet_opt
from common_codes.a4_DE_feature_selection import a4_DE_feature_selection, VAR_NAMES
from common_codes.a5_Fit_lm_model import a5_Fit_lm_model
from common_codes.a7_all_mdl_prediction import a7_all_mdl_prediction
from common_codes.a8_statcalculator import a8_statcalculator
from common_codes.a9_classified_confusion import a9_classified_confusion
from common_codes.a10_compare_all import a10_compare_all
from common_codes.a11_confusion_BlandAltman import a11_confusion_BlandAltman
from common_codes.a12_Predimp import a12_Predimp

import joblib


def WQA_analysis_Jajpur():
    print('=== Water Quality Assessment (WQA) Analysis for Jajpur ===')
    print('Loading data...')

    # Load water quality data
    wqdata = load_wqdata()
    stdwt = load_stdwt()

    if wqdata is None or stdwt is None:
        print("ERROR: wqdata or stdwt not available. Run export_data_to_csv.m in MATLAB first.")
        print("Alternatively, ensure CSV files are in the python_code directory.")
        return

    # Remove TH column only (TDS is not in paper's 12-parameter set)
    for col in ['TH']:
        if col in wqdata.columns:
            wqdata = wqdata.drop(columns=[col])
        if col in stdwt.columns:
            stdwt = stdwt.drop(columns=[col])

    # Get BIS standards array
    BISd = stdwt.values.astype(float)

    # Extract X0 (predictor variables) - paper's 12 hydrochemical params
    # Paper Table 1: pH, EC, DO, F, Cl, NO3, SO4, PO4, U, CaH, MgH, HCO3
    # In wqdata: pH(3), EC(4), DO(5), F(6), Cl(7), NO3(8), SO4(9), PO4(10), U(11), CaH(12), MgH(13), HCO3(14)
    X0 = wqdata.iloc[:, 3:15]
    vars_list = X0.columns.tolist()
    X = X0.values.astype(float)

    # For GWQI: use same 12 parameters
    X_wq = X.copy()

    name = ['GWQI', 'LWQI', 'AWQI', 'RWQI']

    print(f'Data loaded: {X.shape[0]} samples, {X.shape[1]} variables')
    print(f'Variables: {vars_list}')
    print(f'GWQI parameters: {wqdata.columns[3:15].tolist()}')

    # Step a0: Statistics
    print('\n--- a0: Statistics ---')
    Table = a0_statistics_X(X)
    stat_df = pd.DataFrame(Table,
                           columns=['Min', 'Max', 'Mean', 'Std_Dev', 'Skewness', 'Kurtosis'],
                           index=vars_list)
    print(stat_df.round(3))

    # Step a1: Heatmap and boxplot
    print('\n--- a1: Heatmap & Boxplot ---')
    fig1 = a1_heatmap_boxplot(X, vars_list)
    fig1.savefig(os.path.join(_BASE_DIR, 'saved_models', 'figure_a1_heatmap_boxplot.png'), dpi=600, bbox_inches='tight')
    plt.close(fig1)

    # Step a2: GWQI calculation
    print('\n--- a2: GWQI Calculation ---')
    GQ = a2_GWQI(X_wq, BISd)
    print(f'GWQI computed: min={GQ.min():.2f}, max={GQ.max():.2f}, mean={GQ.mean():.2f}')

    # Step a3: Bayesian optimization for Random Forest
    print('\n--- a3: Bayesian Optimization for RF model ---')
    Modells_rf, Opttable_rf = a3_Bayesopt_rf_model(X, GQ)
    print(f'RF model: R2={Opttable_rf["R2"]:.4f}, R2CV={Opttable_rf["R2CV"]:.4f}')
    print(f'Best params: N_estimators={Opttable_rf["NumLearningCycles"]}, '
          f'MinLeaf={Opttable_rf["MinLeafSize"]}, '
          f'MaxSplits={Opttable_rf["MaxNumSplits"]}, '
          f'N_features={Opttable_rf["NumVariablesToSample"]}')

    # Step a4: Bayesian optimization for ANN
    print('\n--- a4: Bayesian Optimization for ANN model ---')
    Modells_ann, Opttable_ann = a4_Bayesian_fitrnet_opt(X, GQ)
    print(f'ANN model (Bayesian): R2={Opttable_ann["R2"]:.4f}, R2CV={Opttable_ann["R2CV"]:.4f}')

    # Step a4b: Differential Evolution for ANN (comparison)
    print('\n--- a4b: Differential Evolution for ANN model ---')
    Modells_ann_de, Opttable_ann_de = a4_DE_fitrnet_opt(X, GQ)
    print(f'ANN model (DE): R2={Opttable_ann_de["R2"]:.4f}, R2CV={Opttable_ann_de["R2CV"]:.4f}')
    print(f'DE best params: Layers={Opttable_ann_de["NumLayers"]}, '
          f'L1={Opttable_ann_de["Layer_1"]}, L2={Opttable_ann_de["Layer_2"]}, '
          f'Act={Opttable_ann_de["Activation"]}, Alpha={Opttable_ann_de["Alpha"]:.6f}')

    # Compare Bayesian vs DE
    print('\n--- Comparison: Bayesian vs DE ---')
    print(f'{"Metric":<20} {"Bayesian":<15} {"DE":<15}')
    print('-' * 50)
    print(f'{"R2":<20} {Opttable_ann["R2"]:<15.4f} {Opttable_ann_de["R2"]:<15.4f}')
    print(f'{"R2CV":<20} {Opttable_ann["R2CV"]:<15.4f} {Opttable_ann_de["R2CV"]:<15.4f}')
    print(f'{"Best Layers":<20} {str(Opttable_ann["NumLayers"]):<15} {str(Opttable_ann_de["NumLayers"]):<15}')
    print(f'{"Best Activation":<20} {Opttable_ann["Activation"]:<15} {Opttable_ann_de["Activation"]:<15}')

    # Step a4b2: CMA-ES for ANN (evolution strategy comparison)
    print('\n--- a4b2: CMA-ES for ANN model ---')
    Modells_ann_cma, Opttable_ann_cma = a4_CMAES_fitrnet_opt(X, GQ)
    print(f'ANN model (CMA-ES): R2={Opttable_ann_cma["R2"]:.4f}, R2CV={Opttable_ann_cma["R2CV"]:.4f}')
    print(f'CMA-ES best params: Layers={Opttable_ann_cma["NumLayers"]}, '
          f'L1={Opttable_ann_cma["Layer_1"]}, L2={Opttable_ann_cma["Layer_2"]}, '
          f'Act={Opttable_ann_cma["Activation"]}, Alpha={Opttable_ann_cma["Alpha"]:.6f}')

    # Three-way comparison: Bayesian vs DE vs CMA-ES
    print('\n--- Comparison: Bayesian vs DE vs CMA-ES ---')
    print(f'{"Metric":<20} {"Bayesian":<15} {"DE":<15} {"CMA-ES":<15}')
    print('-' * 65)
    print(f'{"R2":<20} {Opttable_ann["R2"]:<15.4f} {Opttable_ann_de["R2"]:<15.4f} {Opttable_ann_cma["R2"]:<15.4f}')
    print(f'{"R2CV":<20} {Opttable_ann["R2CV"]:<15.4f} {Opttable_ann_de["R2CV"]:<15.4f} {Opttable_ann_cma["R2CV"]:<15.4f}')
    print(f'{"Best Layers":<20} {str(Opttable_ann["NumLayers"]):<15} {str(Opttable_ann_de["NumLayers"]):<15} {str(Opttable_ann_cma["NumLayers"]):<15}')
    print(f'{"Best Activation":<20} {Opttable_ann["Activation"]:<15} {Opttable_ann_de["Activation"]:<15} {Opttable_ann_cma["Activation"]:<15}')

    # Convergence curve comparison
    print('\n--- Generating convergence curve comparison ---')
    save_dir = os.path.join(_BASE_DIR, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    bayes_conv = Opttable_ann.get('Bayesian_convergence', None)
    de_conv = Opttable_ann_de.get('DE_convergence', None)
    cma_conv = Opttable_ann_cma.get('CMAES_convergence', None)

    if bayes_conv is not None and de_conv is not None:
        de_evals = [c[0] for c in de_conv]
        de_values = [c[1] for c in de_conv]
        cma_evals = [c[0] for c in cma_conv] if cma_conv else []
        cma_values = [c[1] for c in cma_conv] if cma_conv else []

        fig_conv, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(bayes_conv) + 1), bayes_conv, 'b-', linewidth=1.5,
                label=f'Bayesian Optimization (final R²CV={Opttable_ann["R2CV"]:.4f})')
        ax.plot(de_evals, de_values, 'r-', linewidth=1.5, marker='.', markersize=4,
                label=f'Differential Evolution (final R²CV={Opttable_ann_de["R2CV"]:.4f})')
        if cma_conv:
            ax.plot(cma_evals, cma_values, 'g-', linewidth=1.5, marker='s', markersize=4,
                    label=f'CMA-ES (final R²CV={Opttable_ann_cma["R2CV"]:.4f})')
        ax.set_xlabel('Number of Evaluations', fontsize=12)
        ax.set_ylabel('Best R²CV so far', fontsize=12)
        ax.set_title('Convergence Comparison: Bayesian vs DE vs CMA-ES', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.90, 1.005])
        fig_conv.tight_layout()
        fig_conv.savefig(os.path.join(save_dir, 'figure_convergence_comparison.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_conv)
        print(f'  Convergence plot saved: {os.path.join(save_dir, "figure_convergence_comparison.png")}')
    else:
        print('  Convergence data not available, skipping plot.')

    # Step a4c: Regularized MLP with DE (L1 + L2 + Dropout via PyTorch)
    print('\n--- a4c: Regularized MLP with DE (L1+L2+Dropout) ---')
    Modells_ann_reg, Opttable_ann_reg = a4_DE_fitrnet_opt_reg(X, GQ)
    print(f'ANN model (DE-Regularized): R2={Opttable_ann_reg["R2"]:.4f}, R2CV={Opttable_ann_reg["R2CV"]:.4f}')
    print(f'Best params: Layers={Opttable_ann_reg["NumLayers"]}, '
          f'L1={Opttable_ann_reg["Layer_1"]}, L2={Opttable_ann_reg["Layer_2"]}, '
          f'Act={Opttable_ann_reg["Activation"]}, '
          f'L2_reg={Opttable_ann_reg["Alpha_L2"]:.6f}, '
          f'L1_reg={Opttable_ann_reg["Alpha_L1"]:.6f}, '
          f'Dropout={Opttable_ann_reg["Dropout"]:.3f}')

    # Four-way comparison: Bayesian vs DE vs CMA-ES vs DE-Reg
    print('\n--- Four-Way Comparison ---')
    print(f'{"Metric":<25} {"Bayesian":<15} {"DE (L2)":<15} {"CMA-ES":<15} {"DE-Reg":<15}')
    print('-' * 85)
    print(f'{"R2":<25} {Opttable_ann["R2"]:<15.4f} {Opttable_ann_de["R2"]:<15.4f} {Opttable_ann_cma["R2"]:<15.4f} {Opttable_ann_reg["R2"]:<15.4f}')
    print(f'{"R2CV":<25} {Opttable_ann["R2CV"]:<15.4f} {Opttable_ann_de["R2CV"]:<15.4f} {Opttable_ann_cma["R2CV"]:<15.4f} {Opttable_ann_reg["R2CV"]:<15.4f}')

    # Step a4d: DE Feature Selection (simultaneous feature selection + hyperparameter tuning)
    print('\n--- a4d: DE Feature Selection (simultaneous feature + hyperparam optimization) ---')
    Modells_ann_fs, Opttable_ann_fs = a4_DE_feature_selection(X, GQ)
    kept_vars_str = ', '.join(Opttable_ann_fs['KeptVars'])
    dropped_vars_str = ', '.join(Opttable_ann_fs['DroppedVars']) if Opttable_ann_fs['DroppedVars'] else 'none'
    print(f'ANN model (DE-FeatureSelection): R2={Opttable_ann_fs["R2"]:.4f}, R2CV={Opttable_ann_fs["R2CV"]:.4f}')
    print(f'Kept features ({Opttable_ann_fs["N_Features"]}): {kept_vars_str}')
    print(f'Dropped features: {dropped_vars_str}')

    # Four-way comparison including DE-FS
    print('\n--- Four-Way Comparison (including DE Feature Selection) ---')
    print(f'{"Metric":<30} {"Bayesian":<12} {"DE (L2)":<12} {"DE-Reg":<12} {"DE-FS":<12}')
    print(f'{"All Features":<30} {"12":<12} {"12":<12} {"12":<12} {str(Opttable_ann_fs["N_Features"]):<12}')
    print('-' * 78)
    print(f'{"R2":<30} {Opttable_ann["R2"]:<12.4f} {Opttable_ann_de["R2"]:<12.4f} {Opttable_ann_reg["R2"]:<12.4f} {Opttable_ann_fs["R2"]:<12.4f}')
    print(f'{"R2CV":<30} {Opttable_ann["R2CV"]:<12.4f} {Opttable_ann_de["R2CV"]:<12.4f} {Opttable_ann_reg["R2CV"]:<12.4f} {Opttable_ann_fs["R2CV"]:<12.4f}')

    # Save DE-FS results
    joblib.dump(Modells_ann_fs, os.path.join(save_dir, 'Modells_ann_fs.pkl'))
    np.savez(os.path.join(save_dir, 'Opttable_ann_fs.npz'), **Opttable_ann_fs)


    # Use DE-Reg model as the primary ANN for downstream tasks (best generalization)
    # Note: DE-FS model uses a feature subset, so we use DE-Reg for full-feature downstream tasks
    X_ann_subset = X[:, Opttable_ann_fs['FeatureMask']] if 'FeatureMask' in Opttable_ann_fs else X
    Modells_ann = Modells_ann_reg
    Opttable_ann = Opttable_ann_reg

    # Step a5: Linear model
    print('\n--- a5: Linear Model ---')
    Modells_flm, performance_lm = a5_Fit_lm_model(X, GQ)
    print(f'LM model:\n{performance_lm.round(4)}')

    # Save models
    joblib.dump(Modells_flm, os.path.join(save_dir, 'Modells_flm.pkl'))
    joblib.dump(Modells_ann, os.path.join(save_dir, 'Modells_ann.pkl'))
    joblib.dump(Modells_ann_de, os.path.join(save_dir, 'Modells_ann_de.pkl'))
    joblib.dump(Modells_ann_cma, os.path.join(save_dir, 'Modells_ann_cma.pkl'))
    joblib.dump(Modells_ann_reg, os.path.join(save_dir, 'Modells_ann_reg.pkl'))
    joblib.dump(Modells_rf, os.path.join(save_dir, 'Modells_rf.pkl'))
    np.savez(os.path.join(save_dir, 'Opttable_rf.npz'), **Opttable_rf)
    np.savez(os.path.join(save_dir, 'Opttable_ann.npz'), **Opttable_ann)
    np.savez(os.path.join(save_dir, 'Opttable_ann_de.npz'), **Opttable_ann_de)
    np.savez(os.path.join(save_dir, 'Opttable_ann_cma.npz'), **Opttable_ann_cma)
    np.savez(os.path.join(save_dir, 'Opttable_ann_reg.npz'), **Opttable_ann_reg)

    # Step a7: All model predictions
    print('\n--- a7: All Model Predictions ---')
    LQ, AQ, RQ = a7_all_mdl_prediction(X, Modells_flm, Modells_ann, Modells_rf)
    WQIs = np.column_stack([GQ, LQ, AQ, RQ])
    print(f'Predictions: LWQI range=[{LQ.min():.2f}, {LQ.max():.2f}], '
          f'AWQI range=[{AQ.min():.2f}, {AQ.max():.2f}], '
          f'RWQI range=[{RQ.min():.2f}, {RQ.max():.2f}]')

    # Step a8: Performance statistics
    print('\n--- a8: Performance Statistics ---')
    Res = a8_statcalculator(WQIs)
    print(Res.round(4))

    # Step a9: Classified confusion
    print('\n--- a9: Classified Confusion ---')
    WQIclass = a9_classified_confusion(WQIs)

    # Step a10: Comparison plot
    print('\n--- a10: Comparison Plot ---')
    fig10 = a10_compare_all(WQIs, name)
    fig10.savefig(os.path.join(save_dir, 'figure_a10_compare_all.png'), dpi=600, bbox_inches='tight')
    plt.close(fig10)

    # Step a11: Confusion matrix and Bland-Altman
    print('\n--- a11: Confusion & Bland-Altman ---')
    fig11 = a11_confusion_BlandAltman(WQIs)
    fig11.savefig(os.path.join(save_dir, 'figure_a11_confusion_BA.png'), dpi=600, bbox_inches='tight')
    plt.close(fig11)

    # Step a12: Predictor importance
    print('\n--- a12: Predictor Importance ---')
    fig12 = a12_Predimp(X, GQ, vars_list, Modells_flm, Modells_ann, Modells_rf)
    fig12.savefig(os.path.join(save_dir, 'figure_a12_predimp.png'), dpi=600, bbox_inches='tight')
    plt.close(fig12)

    # Print final results
    print('\n' + '='*50)
    print('            Model Performance Summary')
    print('='*50)
    print(Res.round(4))
    print('='*50)
    print(f'\nAll figures saved to: {save_dir}')
    print('Analysis complete!')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    WQA_analysis_Jajpur()