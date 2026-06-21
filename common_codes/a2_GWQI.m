function[QI]=a2_GWQI(X,standard)
valid=sum(~isnan(standard),1)==2;
%%%%%%%%%%%%%%%%%%%%%%%%%
BIS=standard(1,valid);
wt=standard(2,valid);
X=X(:,valid);
%%%%%%
wj=wt/sum(wt);
qj=(X./BIS)*100;

%Column Position of pH entry
pos1=1;pH=X(:,pos1);
pos2=3;DO=X(:,pos2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for pH
% h=pH>7;l=pH<7;
% qpHh=(pH(h)-7)/(8.5-7);qpHl=(pH(l)-7)/(6.5-7);
% %%%%%%%%
% qj(h,pos)=qpHh;qj(l,pos)=qpHl;
qj(:,pos1)=abs((pH-7)/(8.5-7))*100;
qj(:,pos2)=((14.6-DO)/(14.6-5))*100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('');
QI=qj.*wj;
QI=sum(QI,2);