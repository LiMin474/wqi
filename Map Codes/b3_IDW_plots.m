function b3_IDW_plots()
close all
clc
clearvars
%%%%%%%%%%%%%%%%%%%%%%%%%

load geoplott_3.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon
%load geoplott.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon %F, NO3
%load geoplott_2.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon %GQ,LQ,AQ,RQ



cart_map(jprLat,jprLon,mod_Lat,mod_Lon,mod_conc,explat,explon)

disp('')

function cart_map(mypolyLat,mypolyLon,mod_Lat,mod_Lon,conc,explat,explon)
%%Plot the interpolated surface within the shape limits
list={'Cl','SO4','PO4','U','CaH','MgH','HCO3'};
%list={'F','NO3'};
%list={'GQ','LQ','AQ','RQ'};
name='a':'z';
s=figure ();
set(gcf,'color','w');
for j=1:7
Z=conc.(list{j});
subplot(3,3,j)
hold on;
hi=pcolor(mod_Lon,mod_Lat,Z);
b=hi.Parent.YTickLabel;c=strcat(b,'^{\circ}N');hi.Parent.YTickLabel=c;
b=hi.Parent.XTickLabel;c=strcat(b,'^{\circ}E');hi.Parent.XTickLabel=c;
shading flat; %interp;
colormap(jet);
colorbar;
%%%%%%%%%%%%%%%%%%%
pos=hi.Parent.Position;
a = pos(1); b = pos(2); c = pos(3); d = pos(4);
xl=hi.Parent.XLim; yl=hi.Parent.YLim;
p1 = a + (86.3 - xl(1)) / (xl(2) - xl(1)) * c;
q1 = b + (21 - yl(1)) / (yl(2) - yl(1)) * d;
p2 = a + (86.3 - xl(1)) / (xl(2) - xl(1)) * c;
q2 = b + (21.1 - yl(1)) / (yl(2) - yl(1)) * d;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
annotation('textarrow',[p1 p2],[q1 q2],'String','N')
%northarrow('position',[86.7 21.1 1.5 1.5]);
grid on
plot(mypolyLon,mypolyLat,'k'); % Plot the shape boundaries
scatter(explon,explat,10,'filled',"black")
% labels=1:length(explat);
% cellstr(num2str(labels));
% text(explon+0.01,explat+0.01,cellstr(num2str(labels')),'HorizontalAlignment','left')
xlabel('Longitude');
ylabel('Latitude');
title(['(' name(j) ')']);
box on 
grid on
hold off
end
copygraphics(s,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on') 
disp('');





