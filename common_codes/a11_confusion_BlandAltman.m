function a11_confusion_BlandAltman(data)

edges = [0 50 100 150 200 inf];
labels = [1,2, 3, 4, 5];
WQIclass = discretize(data, edges, labels);

s=figure()
set(gcf,'color','w')
name='a':'z';

for i=2:size(WQIclass,2)
    g1=WQIclass(:,1);
    g2=WQIclass(:,i);
C = confusionmat(g1,g2);
subplot(2,3,i-1)
h=confusionchart(C);
h.Title=['(' name(i-1) ')'];
h.RowDisplayLabels={"Excellent","Good"};
h.ColumnDisplayLabels={"Excellent","Good"};
end
[~,n]=size(data);
count=3;
for j=2:n
    count=count+1;
    subplot(2,3,count)
    data1=data(:,1);
    data2=data(:,j);
    data_mean = mean([data1,data2],2);  % Mean of values from each instrument
    data_diff = data1 - data2;              % Difference between data from each instrument
    md = mean(data_diff);               % Mean of difference between instruments
    sd = std(data_diff);                % Std dev of difference between instruments

    h1=scatter(data_mean,data_diff);   % Bland Altman plot
    %h1.Parent.YLim= [-20 20];
    yline(md,'-k');

    yline([1.96*sd,-1.96*sd],['--' 'r']);
    h1.Parent.Title.String=['(' name(count) ')'];
    h1.Parent.XLabel.String='Mean of two measures ';
    h1.Parent.YLabel.String='Diff. in methods ';
    h1.Parent.Box='on';
end
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on')
disp('')
