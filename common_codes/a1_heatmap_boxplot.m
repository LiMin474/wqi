function a1_heatmap_boxplot(X,vars)
s=figure();
set(gcf,'color','w')
subplot(1,2,1)
C=round(corr(X),3);
h=heatmap(C);
vars{6}='NO^{-}_{3}';vars{4}='F^{-}';vars{5}='Cl^{-}';vars{7}='SO^{-2}_{4}';vars{8}='PO^{-3}_{4}';vars{12}='HCO^{-}_{3}';
h.XDisplayLabels=vars(:);
h.YDisplayLabels=vars(:);
h.NodeChildren(3).XAxis.TickLabelInterpreter = 'tex';
h.NodeChildren(3).YAxis.TickLabelInterpreter = 'tex';
h.CellLabelColor='auto'; %;%%'auto'
h.Colormap=jet(5);
h.GridVisible = 'on';
% h.XLabel='Quality';
% h.YLabel='Quality';
title('(a)');

subplot(1,2,2)
[zWQ,~,~]=zscore(X);
id={'pH','EC','DO','F^{-}','Cl^{-}','NO_{3}^{-}','SO_{4}^{2-}','PO_{4}^{3-}','U','Ca H','Mg H','HCO_{3}^{-}'};
if numel(vars)==numel(id)
    vars=id;
end
boxplot(zWQ,'orientation', 'horizontal')
set(gca,'TickLabelInterpreter', 'tex');
set(gca,'YTickLabel',vars,'Fontsize', 11)
title('(b)');
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on');
disp('')