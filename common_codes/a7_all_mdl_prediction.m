function [LQ,AQ,RQ]=a7_all_mdl_prediction(X)

load Fitlm_model.mat Modells_flm
load fitrnet_bayesian.mat Modells_ann
load random_forest.mat Modells_rf
LQ=predict(Modells_flm,X);
AQ=predict(Modells_ann,X);
RQ=predict(Modells_rf,X);

