function a1_bootstrap % Solving the system, check
close all
clearvars
clc
format short g, format compact
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes_today')

nB=100; %% number of Boot samples

load b0_X_GQ.mat GQ X0

load Fitlm_model.mat Modells_flm
load fitrnet_bayesian.mat Modells_ann
load random_forest.mat Modells_rf Opttable_rf
Mdl={Modells_flm Modells_ann Modells_rf};


nResp=numel(Mdl);


Boot_main=cell(1,nResp);


for j= 1:size(Mdl,2)
    XX=X0{:,:};
    YY=GQ;

    nc=size(XX,1);
    ns=size(XX,1);

    YB_sample=zeros(ns,nB);
    RMSE_Boots=zeros(1,nB);


    rng("default")
    for i=1:nB
        idx = randsample(1:nc, round(0.9*nc));%% (nc,nc,true) is with replacement
        Xdata=XX(idx,:);
        Ydata=YY(idx,:);
        boot_model=createmdl(j,Xdata,Ydata,Mdl,Opttable_rf);
        RMSE=find_RMSE(boot_model,Xdata,Ydata);
        RMSE_Boots(1,i)=RMSE;
        Yb_sample=predict(boot_model,XX);
        YB_sample(:,i)=Yb_sample;
    end
   
    Boot_main(1,j)={YB_sample};

end

save Bootconf.mat Boot_main %boot_RMSE
function RMSE=find_RMSE(mdl,X,Y)
pred=predict(mdl,X);
AE=abs(Y-pred);
SE=AE.^2;
MSE=mean(SE,1,"default");
RMSE=MSE.^0.5;

function mdl=createmdl(j,Xdata,Ydata,Mdl,Opttable_rf)
if j==1
    mdl = fitlm(Xdata, Ydata,Mdl{j}.Formula); % Fit a linear model for the first case
end
if j==2
    mdl=fitrnet(Xdata,Ydata,'InitialStepSize','auto','Activations',Mdl{j}.Activations,'Standardize',1,'LayerSizes',Mdl{j}.LayerSizes);
end
if j==3
    para1={'NumVariablesToSample',Opttable_rf.NumVariablesToSample,'MinLeafSize',Opttable_rf.MinLeafSize,'MaxNumSplits',Opttable_rf.MaxNumSplits};
    para2={'Method',Opttable_rf.Method,'NumLearningCycles',Opttable_rf.NumLearningCycles};
    para3={'PredictorSelection','interaction-curvature','Reproducible',true,'Surrogate','on','Type','regression'};
    rng(1)
    t = templateTree(para1{:},para3{:});
    rng(1)
    mdl=fitrensemble(Xdata,Ydata, 'Learners',t,para2{:});
end


