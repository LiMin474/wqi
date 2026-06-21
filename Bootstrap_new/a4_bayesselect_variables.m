function a4_bayesselect_variables % Solving the system, check
close all
clearvars
clc
format short g, format compact
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load b0_X_GQ.mat X0 GQ
X=X0{:,:};
load RF_bayesian_predictor_result.mat mytable

table1=mytable(1:20,:);table2=mytable(21:40 ,:); table3=mytable(41:60 ,:); table4=mytable(61:80 ,:); table5=mytable(81:100 ,:);
table={table1 table2 table3 table4 table5};

for i=1:5
tab=sortrows(table{i},"CVR2","descend");
z=tab(1,:);
ind=z{1,1:12}=='true';
para=z{1,17:18};
features=X0.Properties.VariableNames(:,ind);
ret_feature{i}= features;
parameters{i}=para;
end
save slected_features.mat ret_feature parameters
disp('')
