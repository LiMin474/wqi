function a5_Fit_lm_model(Pred, Resp)
%Pred=normalize(Pred,'center','mean');
%tb=coding(Pred);

type={'linear','interactions'};
columnNames = {'Model', 'DF', 'SumSqr','F-value','p-value','R2','adj-R2','AIC','AICc','R2CV'};
performance = table('Size', [0, numel(columnNames)], 'VariableNames', columnNames,...
    'VariableTypes',['string',repmat({'double'},1,9)]);
for i=1:1
    Modells_flm=fitlm(Pred,Resp,type{i});
    R2CV=crosst(Pred,Resp,type{i});
    tab=anova(Modells_flm,'summary');
    res={type{i},tab.DF(2),tab.SumSq(2),tab.F(2),tab.pValue(2),Modells_flm.Rsquared.Ordinary,...
        Modells_flm.Rsquared.Adjusted,Modells_flm.ModelCriterion.AIC,Modells_flm.ModelCriterion.AICc,R2CV};
    performance(i,:)=res;
end

save Fitlm_model.mat Modells_flm performance
disp('');



function R2CV=crosst(Pred,Yi,mtype)
cv = cvpartition(height(Pred),'KFold',5);
yPredCV = zeros(height(Pred),1);
for k = 1:cv.NumTestSets
    trainIdx = training(cv,k);
    testIdx = test(cv,k);
    mdl_cv = fitlm(Pred(trainIdx,:), Yi(trainIdx,:), mtype);
    yPredCV(testIdx) = predict(mdl_cv, Pred(testIdx, :));
end
SSE=sum((yPredCV-Yi).^2);
SST=sum((mean(Yi)-Yi).^2);
R2CV=1-(SSE/SST);

function coded=coding(tab)
format short
data=tab(:,:);
mini=min(data,[],1);
maxi=max(data,[],1);
data=(2*((data-mini)./(maxi-mini))-1)*1.633;
coded=round(data,3);

