function b1_sampling_plot (filename,wqdata)
% close all
% clear var
% clc, 
% format short G, format compact, profile on; warning('on','all');
% 
% %% The location to the dbf file folder path to be provided
 % filename="ODISHA_SUBDISTRICT_BDY.dbf"; 
 % load a0_Postmonsoon_JAJAPUR.mat  wqdata

GT0 = readgeotable(filename);
GT = GT0(GT0.District == "J>JAPUR",:);
SS=shaperead(filename,'Selector',{@(v1) (strcmp(v1,'J>JAPUR')),'District'});
info = shapeinfo(filename);
proj = info.CoordinateReferenceSystem;
cord={SS.BoundingBox};
pos=[];
for i=1:numel(cord)
    xy=cord{i};
[lati,loni] = projinv(proj,xy(:,1),xy(:,2));
pos=[pos;[mean(lati) mean(loni)]];
end

%BDB=reshape(mean([SS.BoundingBox],1),2,[]);
Blockname={SS.TEHSIL};
Blockname=strrep(Blockname, 'J>JAPUR','JAJAPUR');
figure()
set(gcf,'color','w')
geoplot(GT.Shape,"LineWidth",1,'FaceColor','white','FaceAlpha',1);
hold on
geoscatter(wqdata,"Latitude","Longitude","filled","MarkerEdgeColor","b",MarkerFaceColor="red");
set(gcf,'color','w')
text(pos(:,1), pos(:,2), Blockname, 'Color', 'b', 'FontSize', 10,'HorizontalAlignment','center');

annotation('textarrow',[0.8 0.8],[0.85 0.90],'String','N')
labels=1:height(wqdata);
for i = 1:length(labels)
    text(wqdata.Latitude(i), wqdata.Longitude(i), num2str(labels(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
geobasemap topographic
disp('');
copygraphics(gca,'Resolution',600,'BackgroundColor','white','Padding','tight','PreserveAspectRatio','on') 
disp('');



