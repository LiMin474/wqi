function a2_boot_figure % Solving the system, check
close all
clearvars
clc
format short g, format compact
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes_today\common_codes')
load b0_X_GQ.mat GQ X0
[pred1,pred2,pred3]=a7_all_mdl_prediction(X0{:,:});    % pred 1,2,3 are linear,ann and RF respectively

pred_all=[pred1,pred2,pred3];
a=1:size(pred_all,1);

load Bootconf.mat Boot_main
std_lin=std(Boot_main{1},[],2);
std_ann=std(Boot_main{2},[],2);
std_rf=std(Boot_main{3},[],2);

std_all=[std_lin std_ann std_rf];

s=figure();
set(gcf,'color','w')
name='a':'z';


for i=1:size(pred_all,2)
    subplot(1,3,i)
    h=scatter(a',pred_all(:,i),"o"); hold on
    er_bar= errorbar(a',pred_all(:,i),std_all(:,i), 'vertical', 'LineStyle', 'none','LineWidth',1);
    h.Parent.Title.String=['(' name(i) ')'];
    h.Parent.XLabel.String='Sample ID';
    h.Parent.YLabel.String='Model predicted GWQI';
    box on
end
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on')
disp('')
