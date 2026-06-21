function c3_figure_feature_selection
close all
clearvars
clc
format short g, format compact
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ("D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes_today")
load random_forest.mat Opttable_rf
load fitrnet_bayesian.mat Opttable_ann

load RF_bayesian_predictor_result.mat mytable
resp_rf=mytable;
load Fitrnet_bayesian_predictor_result.mat mytable
resp_ann=mytable;
str.resp1= resp_ann;
str.resp2=resp_rf;

final_A=[sum(resp_ann{:,1:12} == "true",2) resp_ann{:,["R2","CVR2"]}];
fin_A=sortrows(final_A,3,'descend');
[~, id1] = unique(fin_A(:,1),'sorted','first');
ANN_beys=fin_A(id1,:);

final_R=[sum(resp_rf{:,1:12} == "true",2) resp_rf{:,["R2","CVR2"]}];
fin_R=sortrows(final_R,3,'descend');
[~, id2] = unique(fin_R(:,1),'sorted','first');
RF_beys=fin_R(id2,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load features_ppe.mat myT_RF myT_ann
PPE_rf=myT_RF; PPE_ann=myT_ann;
% PPE_rf.variables = cellfun(@(x) x(:).', PPE_rf.variables, 'UniformOutput', false);
% PPE_ann.variables = cellfun(@(x) x(:).', PPE_ann.variables, 'UniformOutput', false);
ANN_PPE=PPE_ann{:,[3 5 6]};
RF_PPE=PPE_rf{:,[3 5 6]};

load features_mrmr.mat myT_RF myT_ann
ANN_MR=myT_ann{:,[3 5 6]};
RF_MR=myT_RF{:,[3 5 6]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Sequential.mat rf_history ann_history
size=cellfun(@numel, rf_history(:,1));
    R2cv=cell2mat(rf_history(:,3));
    R2=cell2mat(rf_history(:,2));
    RF_seq=[size R2 R2cv];

    size_ann=cellfun(@numel, ann_history(:,1));
    R2cv_ann=cell2mat(ann_history(:,3));
    R2_ann=cell2mat(ann_history(:,2));
    ann_seq=[size_ann R2_ann R2cv_ann];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RFadd=[12 Opttable_rf.R2 Opttable_rf.R2CV];
ANNadd=[12 Opttable_ann.R2 Opttable_ann.R2CV];
models={ANN_beys,ANN_MR,ANN_PPE,[RF_beys;RFadd],RF_MR,RF_PPE,[ANNadd;ann_seq],[RFadd;RF_seq]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotscatter(models)
rept=20;
plotheatmap (str,rept)
 disp('')

function plotheatmap(mytable,rept)
ls=fieldnames(mytable);
gg=figure();
set(gcf,'color','w')
name='abcd';
for i=1:length(ls)

    A= mytable.(subsref(ls,substruct('{}',{i})));
    A=A(1:rept,:);
    B = sortrows(A,'CVR2','descend') ;
    X=double(B{:,1:12});
    X(X==2)=0;
    CVR=B.CVR2;


    subplot(1,2,i)
    h=heatmap(X);
    h.XDisplayLabels="v_{"+ string(1:12) +"}";
    y1=(round(CVR,2));
    show=1:round(numel(y1)/10):numel(y1);
    b=nan(size(y1));
    b(show)=y1(show);
    h.YDisplayLabels=b;
    h.NodeChildren(3).XAxis.TickLabelInterpreter = 'tex';
    h.NodeChildren(3).YAxis.TickLabelInterpreter = 'none';
    h.Title= ['(' name(i) ')'];
    h.CellLabelColor='none'; %;%%'auto'
    h.ColorbarVisible='off';
    h.XLabel='Predictors';
    h.YLabel='R^{2}_{CV}';
end
copygraphics(gg,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on') 
disp(' ');

function plotscatter(models)
titles='a':'z';
ff=figure();
set(gcf,'color','w');
for i=1:numel(models)
    % if i >= 4
    %    subplot(2,4,i+1)
    % else
    subplot(2,4,i)
    % end
    tb=models{i};
    h=plot(tb(:,1),tb(:,3),"-o"); 
    h.Parent.XDir="reverse";
    h.Parent.YLim=[0.6 1.1];
    hold on
    plot(tb(:,1),tb(:,2),"-o");
    h.Parent.XLabel.String="Feature set size";
    h.Parent.YLabel.String="Model performance";
    h.Parent.Title.String=['(' titles(i) ')'];
    legend( 'R^2_{cv}','R^2', 'Location', 'Best');
    hold off
end
copygraphics(ff,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on') 
disp(' ');