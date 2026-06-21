function c1_Feature_ppe_mrmr_seq (X0,GQ)
addpath ("D:\Dropbox\Naresh_Sahu\Post Monsoon ANN\Codes_today")
load PPE_mean.mat meanloss

[idxmrmr,~] = fsrmrmr(X0,GQ);
[~, id_ann]= sort(meanloss{2},'descend');
[~, id_RF]= sort(meanloss{3},'descend');


mysequential(X0,GQ);

feature_selection_ppe_mrmr(idxmrmr,X0,GQ,'mrmr','features_mrmr.mat')
feature_selection_ppe_mrmr([id_ann id_RF],X0,GQ,'ppe','features_ppe.mat')

disp('')


function mysequential(Xtr,Ytr)
output=c2_Bayesopt_rf_for_varselect(Xtr,Ytr);
R2cvf=output{2};
n=size(Xtr,2);selected=1:n;
% R2cvf=0.9215;
% selected=[1 2 4 11 12];

R2cvi=0;
filename='Sequential.mat';

rf_history={}; %#ok<NASGU>
while R2cvf>0.90*R2cvi
    count=numel(selected);
    result=cell(count,3);tempMod=cell(count,1);
    for i=1:count
        v=selected;
        v(i)=[];
        %rng(1);

        %output=c2_Bayesopt_ann_for_varselect(Xtr(:,v),Ytr);
        [output]=c2_Bayesopt_rf_for_varselect(Xtr(:,v),Ytr);

        %mdl_R2=output{1};mdl_CV_R2=output{2};mdl=output{3};
        result(i,:)={v,output{1:2}};tempMod(i,:)=output(3);
    end
    [~, il]=sort([result{:,3}],'descend');%% sorted cell array on CVt
    result=result(il,:); %% sorted
    tempMod=tempMod(il,:);
    Newv=result{1,1};newR2=result{1,2};newCVR2=result{1,3};newMdl=tempMod(1,1);
    if newCVR2>0.90*R2cvf
        R2cvi=R2cvf;
        R2cvf=newCVR2;
        selected=Newv;
        newHistory={selected,newR2,newCVR2,newMdl};
        if isfile(filename)
            if ismember('rf_history', who('-file', filename))
                load (filename,'rf_history')
                rf_history=[rf_history;newHistory];
            else
                rf_history=newHistory;
            end
            save (filename,'rf_history','-append')
        else
            rf_history=newHistory;
            save (filename,'rf_history')
        end
    else
        break
    end
end


function feature_selection_ppe_mrmr(ppe,X0,GQ,Method,filename)

if size(ppe,2)==2
    id_ann=ppe(:,1);
    id_RF=ppe(:,2);
else
    id_ann=ppe;
    id_RF=ppe;
end



if exist(filename,'file')==2
    load (filename,'myT_RF','myT_ann')
else
    varNames={'Method','Model','Size','variables','R2','R2CV'};
    myT_RF = table('Size',[0 6],'VariableNames',varNames,'VariableTypes',{'cell','cell','double','cell','double','double'});
    myT_ann = table('Size',[0 6],'VariableNames',varNames,'VariableTypes',{'cell','cell','double','cell','double','double'});
end

for i=1:10

    vars_ann=id_ann(1:2+i); %% starts from top 3 variabes and increasing
    [output_ann]=c2_Bayesopt_ann_for_varselect(X0{:,vars_ann},GQ);
    R2_ann=output_ann{1};R2CV_ann=output_ann{2};Mdl_ann=output_ann{3};
    newRow_ann = {Method,{Mdl_ann}, numel(vars_ann), {vars_ann}, R2_ann, R2CV_ann};
    myT_ann=[myT_ann;newRow_ann]; %#ok<AGROW>

    vars_RF=id_RF(1:2+i); %% starts from top 3 variabes and increasing
    [output_rf]=c2_Bayesopt_rf_for_varselect(X0{:,vars_RF},GQ);
    R2_rf=output_rf{1};R2CV_rf=output_rf{2};Mdl_rf=output_rf{3};
    newRow_rf = {Method,{Mdl_rf},numel(vars_RF), {vars_RF}, R2_rf, R2CV_rf};
    myT_RF=[myT_RF;newRow_rf]; %#ok<AGROW>

    save (filename,'myT_RF', 'myT_ann')
end


