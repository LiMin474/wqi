function b1_variable_selection_main
close all
clearvars
clc
format short g, format compact
profile on
%addpath ('D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes\common_codes')
load b0_X_GQ.mat X0 GQ
X=X0{:,:};
c0_RF_bayesian_20_times (X, GQ)
c0_FITRNET_bayesian_20_times(X,GQ)
c1_Feature_ppe_mrmr_seq(X0,GQ)
c3_figure_feature_selection
c4_stepwise_lm_model(X,GQ);
c6_RF_baysmodel_generate (X,GQ)
c5_model_predict_regression
end