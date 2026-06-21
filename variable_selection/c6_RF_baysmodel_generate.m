function c6_RF_baysmodel_generate (Pred,Resp)

load RF_bayesian_predictor_result.mat mytable
[ma,c]=max(mytable(:,"CVR2"));
mytable=sortrows(mytable,"CVR2","descend");
z=mytable(1,:);
j=c{:,1};
rng(j)
cvs = cvpartition(length(Pred),'KFold',5);
SumSqr(z,Pred,Resp,cvs,j);

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