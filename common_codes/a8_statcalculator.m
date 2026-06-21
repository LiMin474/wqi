function Res=a8_statcalculator(WQIs)
exp=WQIs(:,1); pred=WQIs(:,2:end);
AE=abs((exp-pred));
MAE=mean(AE,1,"default");

% APE=(AE./exp)*100;
% MAPE=mean(APE,1,"omitmissing");

SE=AE.^2;
MSE=mean(SE,1,"default");
RMSE=MSE.^0.5;

% SSE=sum(SE,1,"default");
% SST=sum((exp-mean(exp,1,"default")).^2,1,"default");
% R2=1-(SSE./SST);

Res=[MAE;RMSE];
Res=array2table(Res,"RowNames",{'MAE','RMSE'},"VariableNames",{'LWQI','AWQI','RWQI'});
