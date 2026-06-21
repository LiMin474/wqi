function output=c2_Bayesopt_rf_for_varselect (X,Y)
Mtype="Bag";
nfolds=5;
rng(1)
cvs = cvpartition(size(X,1),'KFold',nfolds);
params=Optopt(X,Y,Mtype);
rng(1)
results = bayesopt(@(paraz)SumSqr(paraz,X,Y,cvs,Mtype,'ita'),params,'ConditionalVariableFcn',@condition,'AcquisitionFunctionName','expected-improvement-plus',...
    'IsObjectiveDeterministic',true,'UseParallel',false,'MaxObjectiveEvaluations',60,'Verbose',1,'PlotFcn',[]);
A1 = results.XAtMinObjective;

[~,output]=SumSqr(A1,X,Y,cvs,Mtype,'final');
disp('');

function X = condition(X)
if ismember('Method', X.Properties.VariableNames)
    X.LearnRate(X.Method ~= 'LSBoost') = NaN;
end
disp('');


function hyperset=Optopt(XX,YY,Modeltype)
hyperset = hyperparameters('fitrensemble',XX,YY,'Tree');
if strcmpi(Modeltype,'Bag')
    hyperset(1).Optimize = false;%'Method'
    hyperset(3).Optimize = false; %'LearnRate'
elseif strcmpi(Modeltype,'LSBoost')
    hyperset(1).Optimize = false;%'Method'
    hyperset(3).Optimize = true; %'LearnRate'
elseif strcmpi(Modeltype,'auto')
    hyperset(1).Optimize = true;%'Method'
    hyperset(3).Optimize = true; %'LearnRate'
end


hyperset(2).Optimize = true; %'NumLearningCycles'
hyperset(2).Range=[10 100];

hyperset(4).Optimize = true; %'MinLeafSize'
%hyperset(4).Range=[1 37]; %[1 37]

hyperset(5).Optimize = true; %'MaxNumSplits'
%hyperset(5).Range=[1 73]; %[2 73]

hyperset(6).Optimize = true;%'NumVariablesToSample'
%hyperset(6).Range=[1 12]; %[1 13]

function [target,output]=SumSqr(z,XX,YY,cvss,Mtype,cond)
if strcmp(Mtype,'auto')
    Name=char(z.Method);
else
    Name=Mtype;
end

para1={'NumVariablesToSample',z.NumVariablesToSample,'MinLeafSize',z.MinLeafSize,'MaxNumSplits',z.MaxNumSplits};
para2={'Method',Name,'NumLearningCycles',z.NumLearningCycles,};
para3={'PredictorSelection','interaction-curvature','Reproducible',true,'Surrogate','on','Type','regression'};
if strcmp(Mtype,'LSBoost')
    para2=[para2,{'LearnRate',z.LearnRate}];
end
rng(1)
t = templateTree(para1{:},para3{:});
Mdl = fitrensemble(XX,YY,'Learners',t,para2{:});
Mdlcv=crossval (Mdl,'CVPartition',cvss);
SST=sum((Mdl.Y-mean(Mdl.Y)).^2);
SSEmdl=resubLoss(Mdl)*numel(Mdl.Y);
SSEcv=kfoldLoss(Mdlcv)*numel(Mdlcv.Y);
R2CV=1-(SSEcv/SST);

if strcmp(cond,'final')
R2 = 1-(SSEmdl/SST);
output={R2,R2CV,Mdl};
target=[];
else
    target=1-R2CV;
    output=[];
end

disp('')

