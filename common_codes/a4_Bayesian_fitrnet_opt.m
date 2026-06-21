function a4_Bayesian_fitrnet_opt (Pred,Resp)

numFolds = 5;
rng (1)
cvs = cvpartition(height(Pred),'KFold',numFolds);



results = bayesopt(@(paraz)SumSqr(paraz,Pred,Resp,cvs),Optopt(),'ConditionalVariableFcn',@condition,'AcquisitionFunctionName','expected-improvement-plus',...
    'IsObjectiveDeterministic',true,'UseParallel',false,'MaxObjectiveEvaluations',60,'Verbose',1,'PlotFcn',[]);
%,'ConditionalVariableFcn',@condition'AcquisitionFunctionName','expected-improvement-plus'
A1=results.XAtMinObjective;
[~,output]=SumSqr(A1,Pred,Resp,cvs);

Mdl=output{3};


A1.Resp=categorical("ANNWQI");A1.R2=output{1};A1.R2CV=output{2};
A1=movevars(A1,'Resp','Before',1);

name='fitrnet_bayesian.mat';
Opttable_ann=A1;
Modells_ann=Mdl;
save (name, 'Opttable_ann', 'Modells_ann')
disp('')



function X = condition(X)
X.Layer_2(X.NumLayers ~= 2) = NaN;
disp('');


function [target,userdata]=SumSqr(z,XX,YY,cvss)
msz=[z.Layer_1 z.Layer_2];
msz=msz(~isnan(msz));

rng (1)
Mdl=fitrnet(XX,YY,'InitialStepSize','auto','Activations',char(z.Activation),'Standardize',1,'LayerSizes',msz);%'Lambda',z.(5),
Mdlcv=crossval(Mdl,'CVPartition',cvss);
SST=sum((Mdl.Y-mean(Mdl.Y)).^2);
SSEmdl=resubLoss(Mdl)*numel(Mdl.Y);
SSEcv=kfoldLoss(Mdlcv)*numel(Mdlcv.Y);
R2CV=1-(SSEcv/SST);
R2 = 1-(SSEmdl/SST);

userdata={R2,R2CV,Mdl};
target=1-R2CV;


function vars=Optopt()
v1= optimizableVariable('NumLayers',[1 2],'Type','integer','Transform','none');
v2= optimizableVariable('Layer_1',[2 10],'Type','integer','Transform','log');
v3= optimizableVariable('Layer_2',[2 10],'Type','integer','Transform','log');
v4= optimizableVariable('Activation',{'tanh' 'sigmoid' 'relu'},'Type','categorical');

%v5= optimizableVariable('Regularization',[8.1301e-08 813.01],'Type','real','Transform','log');
vars=[v1 v2 v3 v4];

