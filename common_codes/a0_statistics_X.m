function Table=a0_statistics_X(X)
avg=mean(X);
MIN=min(X);
MAX=max(X);
Std_Dev=std(X);
Skewn=skewness(X);
kurt=kurtosis(X);
Table=[MIN;MAX;avg;Std_Dev;Skewn;kurt]';
disp('')