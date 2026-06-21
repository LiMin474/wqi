function a3_bayesvarselect_repetation % Solving the system, check
close all
clearvars
clc
format short g, format compact
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes_today\variable_selection')
load b0_X_GQ.mat GQ X0
data=[X0{:,:} GQ];


n = size(data,1);
rng("default");
for i = 1:5
    idx = randperm(n);
    new_data = data(idx(1:round(0.9*n)),:);
    c0_RF_bayesian_20_times(new_data(:,1:12), new_data(:,13))
end
disp('')