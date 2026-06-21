function WQA_analysis_Jajpur
close all
clearvars
clc
format short g, format compact
profile on
addpath(fullfile(pwd, 'common_codes'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load a0_Postmonsoon_JAJAPUR.mat wqdata stdwt
wqdata.TH=[];stdwt.TH=[];wqdata.TDS=[];stdwt.TDS=[];
BISd=stdwt{:,:};
X0=wqdata(:,5:end-4);
vars=X0.Properties.VariableNames;
X=X0{:,:};
name={'GWQI','LWQI','AWQI','RWQI'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Table=a0_statistics_X(X);
a1_heatmap_boxplot(X,vars);
GQ=a2_GWQI(X,BISd);
a3_Bayesopt_rf_model(X,GQ);
a4_Bayesian_fitrnet_opt (X,GQ);
a5_Fit_lm_model(X,GQ);
[LQ,AQ,RQ]=a7_all_mdl_prediction(X); %lm, ANN, RF (baged) predictions with ori data
WQIs=[GQ,LQ,AQ,RQ];
Res=a8_statcalculator(WQIs);
a9_classified_confusion(WQIs);
a10_compare_all(WQIs,name)
a11_confusion_BlandAltman(WQIs)
a12_Predimp(X,GQ,vars);

disp('');

% ====================== 最终最终正确版 ======================
fprintf('\n==================================================\n');
fprintf('              ✔ 模型真实运行结果\n');
fprintf('==================================================\n');
disp(Res);







