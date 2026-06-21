function c0_FITRNET_bayesian_20_times(X,Y) % Solving the system, check
numFolds = 5;
params=Optopt();

check={'ConditionalVariableFcn',@condition,'AcquisitionFunctionName','expected-improvement-plus',...
    'IsObjectiveDeterministic',true,'UseParallel',true,'MaxObjectiveEvaluations',60,'Verbose',1,'PlotFcn',[]};

for j=1:20
    fname='Fitrnet_bayesian_predictor_result.mat';
    rng (j)
    cvs = cvpartition(length(X),'KFold',numFolds);

    results = bayesopt(@(paraz)SumSqr(paraz,X,Y,cvs,j),params,check{:});
    %,'ConditionalVariableFcn',@condition'AcquisitionFunctionName','expected-improvement-plus'
     Tab=results.XAtMinObjective;
    [~,output]=SumSqr(Tab,X,Y,cvs,j);

    Tab=results.XAtMinObjective; Tab.R2=output{1};Tab.CVR2=output{2};

    if exist(fname,'file')~=2
        mytable=Tab;
    else 
        load (fname,'mytable') 
        mytable=[mytable;Tab]; %#ok<AGROW>
    end
    save Fitrnet_bayesian_predictor_result.mat mytable
end

disp('')

function Xnew = condition(X)
Xnew=X;
id=Xnew.(14)==1;
Xnew.(16)(id)=0;
disp('');


function [target,output]=SumSqr(z,XX,YY,cvss,j)
rng(j)
ind=z{1,1:12}=='true';
if z.(14)==1
    msz=z.(15);
else
    msz=[z.(15) z.(16)];
end

if sum(ind)<1
    target=1; output=[];
else
    Mdlcv=fitrnet(XX(:,ind),YY,'CVPartition',cvss,'InitialStepSize','auto','Activations',char(z.(13)),'Standardize',1,'LayerSizes',msz);
    Mdl=fitrnet(XX(:,ind),YY,'InitialStepSize','auto','Activations',char(z.(13)),'Standardize',1,'LayerSizes',msz);
    SST=sum((Mdl.Y-mean(Mdl.Y)).^2);
    SSEcv=kfoldLoss(Mdlcv)*numel(Mdlcv.Y);
    R2CV=1-(SSEcv/SST);

    SSEmdl=resubLoss(Mdl)*numel(Mdl.Y);
    R2 = 1-(SSEmdl/SST);
    output={R2,R2CV,Mdl};
    target=1-R2CV;
end

disp('')



function vars=Optopt()
v1 = optimizableVariable('var1',{'true'  'false'},'Type','categorical');
v2 = optimizableVariable('var2',{'true'  'false'},'Type','categorical');
v3= optimizableVariable('var3',{'true'  'false'},'Type','categorical');
v4= optimizableVariable('var4',{'true'  'false'},'Type','categorical');
v5= optimizableVariable('var5',{'true'  'false'},'Type','categorical');
v6= optimizableVariable('var6',{'true'  'false'},'Type','categorical');
v7= optimizableVariable('var7',{'true'  'false'},'Type','categorical');
v8= optimizableVariable('var8',{'true'  'false'},'Type','categorical');
v9= optimizableVariable('var9',{'true'  'false'},'Type','categorical');
v10= optimizableVariable('var10',{'true'  'false'},'Type','categorical');
v11= optimizableVariable('var11',{'true'  'false'},'Type','categorical');
v12= optimizableVariable('var12',{'true'  'false'},'Type','categorical');

v13= optimizableVariable('Activation',{'tanh' 'sigmoid' 'reLu'},'Type','categorical');
v14= optimizableVariable('NumLayers',[1 2],'Type','integer','Transform','none');
v15= optimizableVariable('Layer_1',[2 10],'Type','integer','Transform','log');
v16= optimizableVariable('Layer_2',[2 10],'Type','integer','Transform','log');
vars=[v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16];


% function [CV_R2,mdl_R2]=calcu(z,XX,YY,cvss,j)
% rng(j)
% ind=z{1,1:12}=='true';
% if z.(14)==1
%     msz=z.(15);
% else
%     msz=[z.(15) z.(16)];
% end
% Mdlcv=fitrnet(XX(:,ind),YY,'CVPartition',cvss,'InitialStepSize','auto','Activations',char(z.(13)),'Standardize',1,'LayerSizes',msz);
% CV_R2=corr(kfoldPredict(Mdlcv),Mdlcv.Y)^2;
% mdlf=fitrnet(XX(:,ind),YY,'InitialStepSize','auto','Activations',char(z.(13)),'Standardize',1,'LayerSizes',msz);
% mdl_R2=corr(predict(mdlf,mdlf.X),mdlf.Y)^2;
% disp('')
