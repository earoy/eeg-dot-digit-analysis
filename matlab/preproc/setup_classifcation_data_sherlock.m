% setup_classification_data_sherlock.m
% ----------------------------------
% Ethan - August 1, 2024
%
% Script to combine pre-processed EEG data and trial level info into
% .mat struct for later classification on sherlock

% edit this line for real data
% to generate updated subjects.txt run in bash: 
%	ls /scratch/users/ethanroy/eeg_groupitizing_data/cleaned/ENI_*_cleaned.mat \
%	>     | xargs -n 1 basename \
%	>     | sed 's/_cleaned\.mat//' \
%	>     | sort \
%	>     > eni_list.txt
%
%	
addpath('/home/users/ethanroy/eeg_groupitizing/code/matlab/preproc/');
% subjectIDs = readcell('/home/users/ethanroy/eeg_groupitizing/code/bash/preproc/eni_list.txt', 'Delimiter','\n');
% Read and decode JSON file
jsonText = fileread('/home/users/ethanroy/eeg_groupitizing/code/bash/preproc/eni_list_test.json');
data = jsondecode(jsonText);

% Extract subject IDs and block numbers
subjectIDs = {data.subjectID}';   % Cell array of IDs
blockNums  = {data.blocks}';      % Cell array of numeric arrays

for s = 1:numel(subjectIDs)
    
    subjectID = subjectIDs{s};
    display('Loading Data for %s', subjectID);

    % Get blocks for this subject
    blocks = blockNums{s};

    % load cleaned EEG data
    eeg_data_path = sprintf("/scratch/users/ethanroy/eeg_groupitizing_data/cleaned/%s_cleaned.mat",subjectID);
    loaded_data = load(eeg_data_path);
    classData.X = loaded_data.xClean;
    classData.good_epochs = loaded_data.INFO.good_epochs;

    % get the directory for behavioral data
    if subjectID=='ENI_032' | subjectID=='ENI_195'
        trial_info_dir = sprintf("/scratch/users/ethanroy/eeg_groupitizing_data/raw_data/%s/Pilot Session 2/Exp Mat Data/", subjectID);
    else
        trial_info_dir = sprintf("/scratch/users/ethanroy/eeg_groupitizing_data/raw_data/%s/ExpMatData/", subjectID);
    end 
    
    outdir = "/scratch/users/ethanroy/eeg_groupitizing_data/classification_input/";
    

    % Get a list of all files in the directory
%     files = dir(fullfile(trial_info_dir, '*.mat')); % Get directory content
%     files = {files.name}; % Extract file names

    files = {};  % Initialize as empty cell array
    
    % loop through the blocks and grab the files
    for b = 1:numel(blocks)
        blockNum = blocks(b);
    
        % Build search pattern for files like *_Block3*.mat
        pattern = sprintf('*_Block%d*.mat', blockNum);
        foundFiles = dir(fullfile(trial_info_dir, pattern));
    
        if isempty(foundFiles)
            warning('No files found for %s Block%d', subjectID, blockNum);
            continue;
        end
    
        % Append to list of files
        files = [files; fullfile({foundFiles.folder}, {foundFiles.name})'];
    end


    % set up .mat object that can then be used with MatclassRSA or Python.
    stim_arr_type = [];
    stim_arr_numerosity = [];
    stim_arr_range = [];
    stim_arr_correct = [];
    
    display('setting up classficiation objects...');
    for i =1:length(files)
        
        file = load(files{i});
        stim_data = file.runOrder;
    
        for j = 2:length(stim_data)
            
            if strcmp(stim_data{j,2}, 'dot')
                stim_cat = 1;
            else
                stim_cat = 2;
            end
          
            stim_arr_type = [stim_arr_type;  stim_cat];
            stim_arr_type = [stim_arr_type;  3];
            stim_arr_type = [stim_arr_type;  stim_cat];
            stim_arr_type = [stim_arr_type;  3];
            stim_arr_type = [stim_arr_type;  2];
    
            stim_arr_numerosity = [stim_arr_numerosity;  stim_data{j,3}];
            stim_arr_numerosity = [stim_arr_numerosity;  100];
            stim_arr_numerosity = [stim_arr_numerosity;  stim_data{j,4}];
            stim_arr_numerosity = [stim_arr_numerosity;  100];
            stim_arr_numerosity = [stim_arr_numerosity;  stim_data{j,5}];
   
            stim_arr_range = [stim_arr_range;  getNumRange(stim_data{j,3})];
            stim_arr_range = [stim_arr_range;  3];
            stim_arr_range = [stim_arr_range;  getNumRange(stim_data{j,4})];
            stim_arr_range = [stim_arr_range;  3];
            stim_arr_range = [stim_arr_range;  getNumRange(stim_data{j,5})];

            stim_arr_correct = [stim_arr_correct; stim_data{j,6}];
    
        end
    end
    
    classData.labels3 = stim_arr_type;
    classData.labels_numerosity = stim_arr_numerosity;
    classData.labels_range= stim_arr_range;
    classData.labels_correct = stim_arr_correct;
    
    display('Saving data...');
    save(fullfile(outdir,subjectID+".mat"), "classData");

end
