function c0_RF_bayesian_20_times (Pred, Resp)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
check={'AcquisitionFunctionName','expected-improvement-plus',...
    'IsObjectiveDeterministic',true,'UseParallel',false,'MaxObjectiveEvaluations',60,'Verbose',1,'PlotFcn',[]};
fname='RF_bayesian_predictor_result.mat';
for j=1:20
    numFolds = 5;
    params=Optopt();
    rng(j)
    cvs = cvpartition(length(Pred),'KFold',numFolds);
    results = bayesopt(@(paraz)SumSqr(paraz,Pred,Resp,cvs,j),params,check{:});%,...
    % 'ConditionalVariableFcn',@condition);
    Tab=results.XAtMinObjective;
    [~,output]=SumSqr(results.XAtMinObjective,Pred,Resp,cvs,j);
    Tab.R2=output{1};Tab.CVR2=output{2};

    if exist(fname,'file')~=2
        mytable=Tab;
    else
        load (fname,'mytable')
        mytable=[mytable;Tab]; %#ok<AGROW>
    end
    save (fname, 'mytable')
end
disp('')



function [target,output]=SumSqr(z,XX,YY,cvss,j)
rng(j)
ind=z{1,1:12}=='true';
if sum(ind)<2
    target=1; output=[];
else
    t = templateTree('MinLeafSize',z.MinLeafSize,'MaxNumSplits',z.MaxNumSplits,'NumVariablesToSample',z.NumVariablesToSample,...
        'PredictorSelection','curvature','Reproducible',true,'Surrogate','on','Type','regression');
    Mdlcv = fitrensemble(XX(:,ind),YY,'NumLearningCycles',z.NumLearningCycles,'Learners',t,'Method','Bag','CVPartition',cvss);%,,
    Mdl = fitrensemble(XX(:,ind),YY,'NumLearningCycles',z.NumLearningCycles,'Learners',t,'Method','Bag');%,,
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
%v13= optimizableVariable('var13',{'true'  'false'},'Type','categorical');

v13= optimizableVariable('MinLeafSize',[1 37],'Type','integer','Transform','log','Optimize',true);
v14= optimizableVariable('MaxNumSplits',[1 73],'Type','integer','Transform','log','Optimize',true);
v15= optimizableVariable('NumVariablesToSample',[1 12],'Type','integer','Transform','log','Optimize',true);
v16= optimizableVariable('NumLearningCycles',[10 100],'Type','integer','Transform','log','Optimize',true);

v17= optimizableVariable('Method',{'Bag'  'LSBoost'},'Type','categorical','Optimize',false);
v18= optimizableVariable('LearnRate',[0.001 1],'Type','real','Transform','log','Optimize',false);



vars=[v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18];



% function X = condition(X)
% ind=X{1,1:12}=='true';
% rep=randsample(1:12,2);
% if sum(ind)<2
%     X{1,rep}={'true'};
% end
% if X.NumVariablesToSample>sum(X{1,1:13}=='true')
%     X.NumVariablesToSample=sum(X{1,1:13}=='true');
% end




% function [CV_R2,mdl_R2]=calcu(z,XX,YY,cvss)
% rng(1)
% ind=z{1,1:12}=='true';
% t = templateTree('MinLeafSize',z.MinLeafSize,'MaxNumSplits',z.MaxNumSplits,'NumVariablesToSample',z.NumVariablesToSample,...
%     'PredictorSelection','curvature','Reproducible',true,'Surrogate','on','Type','regression');
% mdlcv = fitrensemble(XX(:,ind),YY,'NumLearningCycles',z.NumLearningCycles,'Learners',t,'Method','Bag','CVPartition',cvss);%,,
% mdl = fitrensemble(XX(:,ind),YY,'NumLearningCycles',z.NumLearningCycles,'Learners',t,'Method','Bag');%,,
% 
% CV_R2=corr(kfoldPredict(mdlcv),mdlcv.Y)^2;
% mdl_R2=corr(predict(mdl,mdl.X),mdl.Y)^2;
% 
% disp('')