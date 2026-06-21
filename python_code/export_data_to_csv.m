%% Export MATLAB .mat data to CSV files for Python
% Run this script in MATLAB to generate CSV files
% that can be read by the Python data_loader.py
% 
% Usage: Run this script from the project root directory:
%   cd 'D:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu'
%   export_data_to_csv

function export_data_to_csv
    current = pwd;
    save_dir = fullfile(current, 'python_code');
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Export X0 table data from b0_X_GQ.mat
    fprintf('Exporting X0 data...\n');
    load(fullfile(current, 'Bootstrap_new', 'b0_X_GQ.mat'), 'X0', 'GQ');
    writetable(X0, fullfile(save_dir, 'X0_data.csv'));
    dlmwrite(fullfile(save_dir, 'GQ_data.csv'), GQ);

    % Export a0_Postmonsoon_JAJAPUR.mat
    fprintf('Exporting wqdata and stdwt...\n');
    load(fullfile(current, 'a0_Postmonsoon_JAJAPUR.mat'), 'wqdata', 'stdwt');
    wqdata.TH = []; wqdata.TDS = [];
    stdwt.TH = []; stdwt.TDS = [];
    writetable(wqdata, fullfile(save_dir, 'wqdata.csv'));
    writetable(stdwt, fullfile(save_dir, 'stdwt.csv'));

    fprintf('All data exported to %s\n', save_dir);
    disp('Done!');
end