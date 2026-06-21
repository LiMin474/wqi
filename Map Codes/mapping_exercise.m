function mapping_exercise ()
close all
clear var
clc, 
format short G, format compact, profile on; warning('on','all');

%% The location to the dbf file folder path to be provided
samplefilename="ODISHA_SUBDISTRICT_BDY.dbf"; 
load a0_Postmonsoon_JAJAPUR.mat  wqdata
b1_sampling_plot(samplefilename,wqdata)
b2_IDW_extrapolation (wqdata)
b3_IDW_plots()

