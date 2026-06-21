function a12_Predimp(pred,Resp,vars) % Solving the system, check
fname='PPE_mean.mat';

if exist(fname,'file')~=2
    load Fitlm_model.mat Modells_flm
    load fitrnet_bayesian.mat Modells_ann
    load random_forest.mat Modells_rf
    vars{4}='F^{-}';vars{5}='Cl^{-}';vars{6}='NO^{-}_{3}';vars{7}='SO^{-2}_{4}';vars{8}='PO^{-3}_{4}';vars{12}='HCO^{-}_{3}';
    mdl={Modells_flm; Modells_ann; Modells_rf};
    n=numel(mdl);
    loss=cell(1,n);meanloss=cell(1,n);
    stdloss=cell(1,n);Tol=cell(1,n);
    for i = 1:n
        A = MDL_PPE(mdl{i}, pred, Resp, 100);
        B = mean(A,2);
        C = std(A,[],2);
        D = MDL_tol(mdl{i}, pred);
        loss(1,i) ={A};meanloss(1,i)={B};stdloss(1,i)={C};Tol(1,i)={D};
    end
    save (fname, 'loss', 'meanloss', 'stdloss', 'Tol' ,'vars')
else
    load (fname, 'meanloss', 'stdloss', 'Tol' ,'vars')
end
plotPPEsens(meanloss, stdloss,Tol, vars);



function plotPPEsens(meanloss, stdloss, Tol, vars)
s=figure();
set(gcf,'color','w');
tiledlayout(3,2, 'Padding','compact','TileSpacing','compact');
titleLabels = {'(a)','(b)','(c)','(d)','(e)','(f)'};
for i=1:3
    ax1 = nexttile((i-1)*2+1);
    ax2 = nexttile((i-1)*2+2);
    title1=titleLabels{(i-1)*2+1};
    title2=titleLabels{(i-1)*2+2};

    axes(ax1); cla(ax1)
    bar(ax1, reordercats(categorical(vars), vars), meanloss{i});
    xlabel(ax1, 'Predictors')
    ylabel(ax1, 'Increase in model MSE')
    ylim(ax1, [0 75]);
    title(ax1, title1)
    hold on
    errorbar(ax1, categorical(vars), meanloss{i}, stdloss{i}, 'k', 'LineStyle','none');
    ax1.TickLabelInterpreter = 'tex';
    hold off
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    axes(ax2); cla(ax2)
    stem(ax2, Tol{i});
    xlabel(ax2, 'Observations')
    ylabel(ax2, 'Deviation from model (%)')
    ylim(ax2, [-10 10])
    yline(ax2, 5, '--r')
    yline(ax2, -5, '--r')
    legend(ax2, {'Perturbed Model'}, 'Location', 'southwest')
    title(ax2, title2)
end
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on')
disp (' ');

function Tol=MDL_tol(mdl,pred)
A=[];
for i= 1:100
    err=0.95+0.1*rand(size(pred));
    newmat=pred.*err;
    A=[A predict(mdl,newmat)];
end
alt_model=mean(A,2);
Ori_mod=predict(mdl,pred);
Tol=((alt_model-Ori_mod)./Ori_mod)*100;
disp('');

function lossmatrix=MDL_PPE(MDL,X,Y,rept)
num_pred=size(X,2);
lossmatrix=zeros(num_pred,rept);
for i=1:num_pred
    Xtemp=X;
    val=X(:,i);
    for j=1:100
        Xtemp(:,i)=val(randperm(length(val)));
        pred=predict(MDL,Xtemp);
        loss=mean((pred-Y).^2,1);
        lossmatrix(i,j)=loss;
    end
end

