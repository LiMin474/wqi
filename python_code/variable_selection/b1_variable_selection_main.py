import os
import sys


def b1_variable_selection_main(GQ, X0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_dir)

    from variable_selection.c0_RF_bayesian_20_times import c0_RF_bayesian_20_times
    from variable_selection.c0_FITRNET_bayesian_20_times import c0_FITRNET_bayesian_20_times
    from variable_selection.c1_Feature_ppe_mrmr_seq import c1_Feature_ppe_mrmr_seq
    from variable_selection.c3_figure_feature_selection import c3_figure_feature_selection
    from variable_selection.c4_stepwise_lm_model import c4_stepwise_lm_model
    from variable_selection.c6_RF_baysmodel_generate import c6_RF_baysmodel_generate
    from variable_selection.c5_model_predict_regression import c5_model_predict_regression

    X = X0
    c0_RF_bayesian_20_times(X, GQ)
    c0_FITRNET_bayesian_20_times(X, GQ)
    c1_Feature_ppe_mrmr_seq(X0, GQ)
    c3_figure_feature_selection()
    c4_stepwise_lm_model(X, GQ)
    c6_RF_baysmodel_generate(X, GQ)
    c5_model_predict_regression()