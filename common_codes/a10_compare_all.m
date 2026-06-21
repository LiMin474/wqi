function a10_compare_all(All,name)
a=1:size(All,1);
s=figure()
set(gcf,'color','w')
h1=plot(a',All);
baseColors = {'r', 'g', 'b', 'y', 'c', 'm', 'k'};
use=baseColors(1:numel(h1));
[h1.Color]=deal(use{:});%"r","g","b","c","m"
[h1.Marker]=deal('o');
[h1.LineStyle]=deal(':');
[h1.MarkerFaceColor] = deal(use{:});
l=legend(h1);
l.String=cellstr(strcat(name));
l.AutoUpdate="off";
yline([50 100 150],'--g')
xlabel('Sample ID');ylabel('WQI value');
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on')
disp('****')