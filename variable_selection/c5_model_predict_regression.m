function c5_model_predict_regression
close all
clearvars

load RF_mdl_beys.mat output
model=output{:,3};
load b0_X_GQ.mat X0 GQ
X=X0{:,:};
load RF_bayesian_predictor_result.mat mytable
mytable=sortrows(mytable,"CVR2","descend");
z=mytable(1,:);
ind=z{1,1:12}=='true';
pred=predict(model,X(:,ind));
R2=output{:,1};

fig=figure();
set(gcf,'color','w')
ft=12;
subplot(1,2,1)
xylim=[min([GQ,pred],[],'all') max([GQ,pred],[],'all')];
h1=scatter(GQ,pred);
h1.Parent.XLim=xylim;
h1.Parent.YLim=xylim;
h1.MarkerEdgeColor='green';
h1.Parent.XLabel.String="Original GWQI";
h1.Parent.YLabel.String="RF Model WQI";
h1.Parent.TitleHorizontalAlignment="center";
h1.Parent.FontSize=ft;
h1.Parent.Title.String='(a)';
txt = ['R^{2}: ' num2str(round(R2,3,"significant"))];
text(mean(h1.Parent.XLim),0.75*mean(h1.Parent.YLim),txt,"FontSize",ft)
h2=lsline;
h2.Color='red';h2.LineStyle="-";h2.LineWidth=1;
%legend(leg(i),'Location','northwest','Interpreter','none','Box','off')
box 'on'


subplot(1,2,2)
resu=(pred-GQ);
MAE=(resu./std(resu,1)); %% standardized residuals
h=stem(1:numel(GQ),MAE);
h.Color="b";
h.Parent.YLim=[-4 4];
% h(1).Parent.XLim=[0 31];

h.MarkerEdgeColor="g";
h.Parent.XLabel.String='Observations';
h.Parent.YLabel.String='Residuals (standardized)';
yline([-3 3],'--r');

% [h.LineWidth]=deal(1.5);
% h(1).Parent.LineWidth=1.5;
 h.Parent.FontSize=ft;
% h(1).Parent.LabelFontSizeMultiplier=1.2;
h.Parent.Title.String='(b)';
copygraphics(fig,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on') 



disp('')