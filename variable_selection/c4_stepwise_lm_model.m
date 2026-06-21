function c4_stepwise_lm_model(Pred, Resp)

Pred=array2table(Pred);
Resp=array2table(Resp);

    [perf,model]=mdlscreen(Pred,Resp);

save stepwiselm_model.mat model perf
disp('');
end



function [dataTable,megaModel]= mdlscreen(Pred,Yi)
columnNames = {'Model', 'DF', 'SumSqr','F-value','p-value','R2','adj-R2','AIC','AICc','R2CV'};
dataTable = table('Size', [0, numel(columnNames)], 'VariableNames', columnNames,...
    'VariableTypes',{'string','double','double','double','double','double','double','double','double','double'});
megaModel=cell(1,1);
    mdl_lm=stepwiselm(Pred{:,:},Yi{:,:},'constant','Upper','linear');
    rng(1);
    cv = cvpartition(height(Pred), 'KFold', 5);
    yPredCV = zeros(height(Pred), 1);
    for k = 1:cv.NumTestSets
        trainIdx = training(cv,k);
        testIdx = test(cv,k);
        mdl_cv = fitlm(Pred(trainIdx,:), Yi(trainIdx,:), 'constant','Upper','linear');
        yPredCV(testIdx) = predict(mdl_cv, Pred(testIdx, :));
    end
    SSE=sum((yPredCV-Yi).^2);
    SST=sum((mean(Yi)-Yi).^2);
    R2cv=1-(SSE/SST);
    % for k = 1:cv.NumTestSets
    %     trainIdx = training(cv, k);
    %     testIdx = test(cv, k);
    %     mdl_cv = stepwiselm(Pred(trainIdx, :), Yi{trainIdx, :},'constant','Upper','linear');
    %     yPredCV(testIdx) = predict(mdl_cv, Pred(testIdx, :));
    % end
    % R2cv = corr(yPredCV, Yi{:,:})^2;
megaModel(1)={mdl_lm};
tab=anova(mdl_lm,'summary');
data={{'linear'},tab.DF(2),tab.SumSq(2),tab.F(2),tab.pValue(2),mdl_lm.Rsquared.Ordinary,mdl_lm.Rsquared.Adjusted,mdl_lm.ModelCriterion.AIC,mdl_lm.ModelCriterion.AICc,R2cv};
dataTable(1,:)=cell2table(data, 'VariableNames', columnNames);
disp(dataTable)
disp('');
end

