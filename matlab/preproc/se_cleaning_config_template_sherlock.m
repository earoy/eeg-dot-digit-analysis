% se_cleaning_config.m
% -------------------------------------------------------------------------
% By Amilcar Malave (2025-01-16)
% Adapted from: preproc_0_config.m by Blair Kaneshiro (2024-08-07)

% Config file in which personalized and/or fairly stable information (e.g.,
% input/output/figure directories, filtering specifications) can be stored
% in a separate location from the preprocessing code. Some variables and
% fields are declared as empty and are filled in during preprocessing. 

% INSTRUCTIONS FOR USE: 
% - Create your own version of the config file by adding an underscore and
%   your initials at the end of the filename. 
%       Example: se_config_BK.m
% - Git will ignore any config file with an underscore after the word
%   'config'. 
% - Fill in all the variables with your own directory info, etc. Variables
%   and fields which should be declared as empty at this stage are indicated
%   as such. 
% - Any preprocessing *functions* that use this config file will accept the
%   config filename as an input. 
% - Be sure to update any preprocessing *scripts* that use this config file
%   with the name of your specific config file. 
% - The script structure can be viewed by clicking the "Go To" button in 
%   the "EDITOR" tab.

% % * Script history * %

fprintf('Pipeline Configuration\n')

%% Directories
% User should only need to specify items in this section once

% High level
% Path of this Repo
% subjectID = 'Insert subject ID';

INFO.dirs.code = '/home/groups/brucemc/Analysis/SENSI-EEG-Preproc-SE-private/';
 % The name of this file (Change with your config version, E.g shortEpoch_config_BK.m)
INFO.configFn = sprintf('/Users/school/Desktop/DataCleaning/ENI_%s/se_cleaning_config_ENI_%s.m', subjectID);

% Specific preprocessing stages - Input/output/figure directories
 INFO.dirs.raw = sprintf('/scratch/users/ethanroy/eeg_groupitizing_data/raw_data/ENI_%s', subjectID);
 INFO.dirs.cleaned = '/scratch/users/ethanroy/eeg_groupitizing_data/cleaned/';

 INFO.dirs.figs = '/scratch/users/ethanroy/eeg_groupitizing_data/cleaned/';

%% General Parameters

%%% Specify Figure Vizualization/Export/Clear 
config.doFigs = 1; % Do Figures -> 1 = ON, 0 = OFF
config.doFilterFig = 0; % Do filter Figs (take long) -> 1 = ON, 0 = OFF
config.showFigs = 0; % Figure Visibility -> 1 = ON, 0 = OFF
config.saveFigs = 1; % Export Figure -> 1 = ON, 0 = OFF
config.closeFigs = 1; % Close Figures after exporting (reduce RAM use) -> 1 = ON, 0 = OFF
config.fig_res = 300; % figure resolution - Higher is better
config.ePlot = []; % Channel for Plotting (default value if left empty)

%%% Specify Output Exports
config.saveOutput = 1; % ON = 1, OFF = 0;

%%% Free varaibles to safe Ram (Clear Intermediate Variables)
config.freeVariables = 1; % ON = 1, OFF = 0;

%%%%%%%%%%%%%%%%%%% Annotations (fill and/or init) %%%%%%%%%%%%%%%%%%%%%%%
INFO.anno.dataCollectionDate = ''; % Leave empty when initializing this file
INFO.anno.experimenter = ''; % Leave empty when initializing this file
INFO.anno.stimArray = ''; % Leave empty when initializing this file
INFO.anno.net = ''; % Leave empty when initializing this file
INFO.anno.impedancesChecked = 'Y'; % User can overwrite in each run
INFO.anno.participantWearingMask = 'N'; % User can overwrite in each run

%% File search and save strings

% User specifies items in this section for every file set. 

% Participant identifiers
    % Always specify: 
    
    % 1. File Prefix
    INFO.file_labels.Prefix = "ENI"; 
    % INFO.file_labels.Prefix = "S"; 

    
    % 2. Subjects to be analyzed (Cell array) 
    % INFO.file_labels.Subjects = {"001","003"}; 
    % INFO.file_labels.Subjects = {""}; 

    % INFO.file_labels.Subjects = {"006"}; % SSVEP can Only take one at a time
    
    % 3. Block identifier - Select Which Blocks to use
    config.Block_ID = 1;
    
        % 3.1 Blocks to be Analyzed - Costumize your blocks
        % tmp.Block{1} = {"Block_1","Block_2","Block_3","Block_4"};   % Blocks a1-3
        % tmp.Block{1} = {"a1"};   % Blocks a1-3
        % tmp.Block{1} = {"ERP"};   % Block Word
        % tmp.Block{2} = {"b1","b2"};   % Blocks b1-3
        INFO.file_labels.Blocks = tmp.Block{config.Block_ID};
    
        % 3.2 Output filename termination
        % tmp.BlockStr{1} = sprintf("%s", subjectID); 
        tmp.BlockStr{1} = INFO.file_labels.Subjects{1};
        % tmp.BlockStr{2} = "195";
        INFO.file_labels.out_file = tmp.BlockStr{config.Block_ID};
    

%% Type of Study

%%%%%%%%%%%%%%%%%%%%%% Type of Study %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (make the pipeline run the correct lines)

% Type of Onset - DIN or OCED style? 
% Always specify: 
tmp.onset_type_id = 1; % DIN = 1, OCED = 2
tmp.onset_type = {"DIN", "Oced"}; % Onset Types (dont touch this line)

%%% ERP Conditions / Categories
    % When to specify: Only for ERP
    % Desired TCPIP table Conditions / Categories 
    tmp.TCPIP_conditions = {'STIM'}; % TCPIP Table Epoch Conditions

    % OCED labels for input's "categoryLabels"
    tmp.OCED_conditions = {'STIM'}; % OCED Categories

%%% Study Style - ERP or SSVEP
% Always specify: 
tmp.study_style_id = 1; % ERP = 1, SSVEP = 2
tmp.study_style = {"ERP", "SSVEP"}; % Study Style (dont touch this line)
INFO.study_style = tmp.study_style{tmp.study_style_id};

%%%%%%%%%%%%%%%%%%% Custom %%%%%%%%%%%%%%%%%%%%%%%%
% Select Custom Study
% Always specify: turn on only for SSVEP with DIN epochs (TCPIP not in use)
config.Custom.Fang_SSVEP = 0; % ON = 1, OFF = 0;

%%% SSVEP Only (Custom for Fang)
if config.Custom.Fang_SSVEP
    INFO.onsets.din_start = 1; % First DIN for Epoching
    INFO.onsets.ds = 3; % Din Spacing -> ds = 3 leads to DIN (1, 4, 7, ...)
    INFO.onsets.nEpochs = 12; % Number of Epochs per Trial
    INFO.onsets.nTrials = 10; % Number of Trials
    INFO.onsets.epochs_keep = [2 3 4 5 6 7 8 9 10 11]; % Indices of Epochs to keep
    INFO.onsets.trials_keep = [2 3 4 5 6 7 8 9]; % Which trials to keep
    config.plotFFT.window = [0 50]; % Window for FFT plots
    config.plotFFT.plotTrial = 5; % Which Trial subset for FFT plots
    config.plotFFT.desiredFreqs = [1 3]; % Desired Freqz to Plot (Reference lines)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Sampling rate info 

% Expected acquisition sampling rate
% Always specify
INFO.fs_0 = 1000; % Should match the loaded file sampling freq variable


%% Truncate Continuous Recording 
% Automated Implementation - No need to specify:

% % How many msec to truncate of each recording - set to 0 to ignore
% INFO.truncateMsec.Start = 0; % Time to truncate from the start (milliseconds)
% INFO.truncateMsec.End = 100;   % Time to truncate from the end (milliseconds)
% 
% % How many msec to visualize from end of each recording
% INFO.truncateMsec.Window = 5000; % Time Window for truncation plots (milliseconds)


%% Filtering and Resampling 
% Always specify: 

INFO.FILTERING.hpHz = 0.3;              % Highpass cutoff, in Hz
INFO.FILTERING.notchHz = [59 61];       % Notch boundaries, in Hz
INFO.FILTERING.lpHz = 50;               % Lowpass cutoff, in Hz
INFO.FILTERING.filterSpecs = [];        % Filter specs returned by fn call - leave empty when initializing this file

% Resampling Frequency (Hz) - ERP and SSVEP
% Always specify -> It can be the same as fs_0 (no resampling or downsampling)
INFO.FILTERING.Resample = 200;  % IF ERP, fs_0/resample should be a integer    

% Downsampling Factor for ERP (Ensuring DS in an ingeter)
if INFO.study_style == "ERP"
    INFO.FILTERING.DS = round(INFO.fs_0/INFO.FILTERING.Resample); % Integer
    INFO.FILTERING.Resample = INFO.fs_0/INFO.FILTERING.DS; % New Resampling Freq
end

%% Ephoching INFO
% Always specify: 

% Long Epoch - Longer epoch to apply eye regression
INFO.epoch.LStartMsec = -200; % Time bofore onset (for SSVEP, it shoulbe be = 0)
INFO.epoch.LEndMsec = 1000; % Time after onset

% Short Epoch - Desired Epoch Length
% Epoch start/end for output data. Procedure below will include the
INFO.epoch.SStartMsec = -150; % Time bofore onset (for SSVEP, it shoulbe be = 0)
INFO.epoch.SEndMsec = 600; % Time after onset

% Onset Trigger
% Value that all DIN triggers will end up getting
INFO.trigger.inDIN = 3;        % Expected value of input DIN triggers (0=no Din, 1=photodiode, 2=audio click)
INFO.trigger.outDIN = 8888;    % Output value of DIN triggers (use value not represented in TCP set to avoid overlap)

%%%%%%%%%%%%%%  Din Offset  %%%%%%%%%%%%%%% 
% Can modify if Necessary (e.g. the offset does not match)
% Can modify if Necessary (e.g. the offset does not match)
if INFO.trigger.inDIN == 1           % Photodiode
    INFO.onset.dinSampOffset = 0;    % SENSI: Typically 0 samples (0 msec) 
elseif INFO.trigger.inDIN == 2       % Audio
    INFO.onset.dinSampOffset = 1000; % SENSI: Typically 1000 samples (1000 msec)
elseif INFO.trigger.inDIN == 3
    INFO.onset.dinSampOffset = 0;    % SENSI: Typically 1000 samples (1000 msec) 
    disp('DIN3 Onsets - Visual');
end


%% Eye Regression
% Perform Eye Regression
config.do_eye_regression = 1; % ON = 1, OFF = 0;

%%%%%%%%%%%%% Default VEOG and HEOG %%%%%%%%%%%%%%% 
% EGI
EOG.chVEOG = [8 25]; % User can overwrite in each run
EOG.chHEOG = [1 32 125 128]; % User can overwrite in each run

% Easycap
EOG.vChan = [1 3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Reduce Channels 
% Reduce channel montage if using EGI data
config.useElectrodes = 1:124; % Channels to Keep


%% Bad Channel Info 
% Always specify: 

% Perform Bad Channel Fix?
config.fixBadChs = 1; % ON = 1, OFF = 0;

% General Bad Channel Fix
INFO.badCh.medianHighThress = 50; % High Median Voltage Thresshold
INFO.badCh.medianLowThress = 0.5; % Low Median Voltage Thresshold

% Per Epoch Bad Channel Fix
% For each Epoch, remove channel with bad sample % > config.badCh.pctThress
INFO.badCh.UVThres = 100;
INFO.badCh.pctThress = 10; % Remove  Channel if bad samples > thresshold

INFO.bad_channels = {}; % Leave empty when initializing this file

% Avoid Removal / Imputation of the following Channels
config.badCh.refCh = 129; % Reference Channel (Do not remove this one)
config.badCh.keep = [8 25 1 32 125 128]; % Chs to keep intact (row vector)
    % E.g.
    % Custom -> [1,2, 5:14, 18, 19]; 
    % EGI Eye Channels ->  [8 25 1 32 125 128];
    % Empty -> [];

% Adding Reference and ePlots Chs to "badCh.keep" (avoid removing them)
config.badCh.keep = [config.badCh.keep config.badCh.refCh config.ePlot];


%% Bad Samples and Epoch Exclusion

%%%%%%%%%%%%%%%%%%% NaN BAD Samples %%%%%%%%%%%%%%%%%%%%%%%%% 
% NaN Bad Samples: ON = 1, OFF = 0;

% Get NaNs based on Standard Deviation 
config.exclude.do_nan_std = 1; % ON = 1, OFF = 0;

% Get NaNs based on Voltage Threshold 
config.exclude.do_nan_voltageThres = 1; % ON = 1, OFF = 0;
INFO.Thresh.recUV = 100; % Turn Samples > this into NaNs

%%%%%%%%%%%%%%%%% Exclude and/or Fix Bad Epochs/Samples %%%%%%%%%%%%%%%%%%%
% Exclude Bad Epochs
config.exclude_bad_epochs = 0; % ON = 1, OFF = 0;
config.label_bad_epochs = 1;

% Exclude based on the % of NaNs
INFO.Thresh.pctNaN = 10; % More than this percentage of NaNs per epoch
INFO.Thresh.pctSampleNaN = 45; % More than this % of NaNs in a single sample  

%%%%%%%%%%%%%%%%%%% Do Imputation (Remove NaNs) %%%%%%%%%%%%%%%%%%%%%%%%%%
config.Impute = 1; % ON = 1, OFF = 0;


%% Final Clean
% Do Average Reference?
config.doAverageReference = 1; % ON = 1, OFF = 0;


%% Optional

% Float Precesion: single (half memory - faster), double (more precise - slower)
% For "single" - Need to fix the matrix creation of all functions
INFO.Precision = "single"; % String: "single" or "double" 


%% DO NOT TOUCH FROM HERE

%% Initializing Extra Parameters and Permorming Preliminary Checks

%%%%%%%%%%%%%%%%%%%%% Extra - Do not touch from here %%%%%%%%%%%%%%%%%%%%
figSt = struct();
if ~exist('vars', 'var') || ~isstruct(vars)
    vars = struct();
end

%%%%%%%%%%%%%%%%%%%%%% Adding Dirs Paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp.fields = fieldnames(INFO.dirs); % Get all field names

for i = 1:numel(tmp.fields)
    dirPath = INFO.dirs.(tmp.fields{i}); % Get the directory path
    
    if exist(dirPath, 'dir') % Check if folder exists
        addpath(genpath(dirPath)); % Add to path
    else
        % Show a popup message asking to create the folder
        choice = questdlg(sprintf('Folder not found:\n%s\nDo you want to create it?', dirPath), ...
                          'Missing Folder', ...
                          'Yes', 'No', 'Yes'); 
        
        if strcmp(choice, 'Yes')
            mkdir(dirPath); % Create the folder
            addpath(genpath(dirPath)); % Add it to path after creation
            msgbox(sprintf('Folder created:\n%s', dirPath), 'Success', 'help');
        else
            warning('Skipped missing folder: %s', dirPath);
        end
    end
end

%%%%%%% Ensure repo files are in path %%%%%%%%%
assert(~isempty(which('epochContinuousData')), ...
    'Make sure the preprocessing pipeline repo and sub-folders are added to Matlab path.')


clear i dirPath choice;

%% Cleaning Only - Parameters and Preliminary Checks

% Print Sampling Frequency
fprintf('\tInput fs = %gHz\n', INFO.fs_0);

% Downsampling Factor for ERP (Calculated using resampling freq)
if INFO.study_style == "ERP"
    fprintf('\tDownSampling Factor = %g\n', INFO.FILTERING.DS);
end

% Resampling Frequency
fprintf('\tResampling fs = %gHz\n', INFO.FILTERING.Resample);


%%%%%%%%%%%%%%%%%%%%% Steps Checks %%%%%%%%%%%%%%%%%%%%
config.excluded = 0;

%%%%%%% Ensure SSVEP epoch start at zero %%%%%%%%%
if config.Custom.Fang_SSVEP
    if INFO.epoch.LStartMsec ~= 0 && INFO.epoch.SStartMsec ~= 0
        error(sprintf(['For SSVEP studies - Config file:\n' ...
            'INFO.epoch.LStartMse & INFO.epoch.SStartMsec must be = 0']))
    end
end

%%%%%%%%%%%%%%  Din Offset  %%%%%%%%%%%%%%% 
if INFO.trigger.inDIN == 1  % Photodiode
    fprintf('\tDIN1 Onsets - Photodiode, SampOffset = %g ms\n', INFO.onset.dinSampOffset);
elseif INFO.trigger.inDIN == 2 % Audio
    fprintf('\tDIN2 Onsets - Audio, SampOffset = %g ms\n', INFO.onset.dinSampOffset);
else % Oced Style
    fprintf('\tOnsets should be defined in the raw file as "onsets"\n');
end

%%%%%%%%% Avoid Resampling Aliasing %%%%%%%%% 
if INFO.FILTERING.Resample < 2*INFO.FILTERING.lpHz
    error(['Avoid aliasing\nEnsure that %.1f Hz (INFO.FILTERING.Resample) ' ...
        '>= 2*%.1f Hz (2 * low-pass cutoff)'], ...
        INFO.FILTERING.Resample, INFO.FILTERING.lpHz);
end

%%%%%%%%%%%%%%%%% Avoid float Onsets %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

tmpfs_0 = INFO.fs_0;
tmpfs_RS = INFO.FILTERING.Resample;



%% Include in Future Iterations

% Impedances for specific recording(s)
% INFO.impedances = []; % Leave empty when initializing this file

% %%%%%% Percent of data replaced by NaNs in nanBadSamples iterations %%%%%%
% INFO.pctNBS = [];   % Leave empty when initializing this file


%%% Other strings for searching and labelling
% INFO.fSaveStr = 'SE_';               % For output files

% % User-entered search string - will be entered in preproc 1
% INFO.sStr = ''; % Leave empty when initializing this file   
% INFO.fSearchStrUse = {}; % Leave empty when initializing this file


%% Legacy

%%%%%%%%%%%%%% Files loaded; files saved; run datetime %%%%%%%%%%%%%%%%

% % preproc_1a_loadFilterEpochOneFile.m: One or more files in, one file out
% INFO.preproc1_MR_fNamesLoaded = {}; % Leave empty when initializing this file
% INFO.preproc1_MR_nFilesLoaded = []; % Leave empty when initializing this file
% INFO.preproc1_MRE_fNameSaved = []; % Leave empty when initializing this file
% INFO.preproc1_analyzer = []; % Leave empty when initializing this file
% INFO.preproc1_datetime = []; % Leave empty when initializing this file
% 
% % preproc_2_preICA.m: One file in, one file out
% INFO.preproc2_MRE_fNameLoaded = {}; % Leave empty when initializing this file
% INFO.preproc2_MIR_fNameSaved = {}; % Leave empty when initializing this file
% INFO.preproc2_analyzer = []; % Leave empty when initializing this file
% INFO.preproc2_datetime = []; % Leave empty when initializing this file

% % preproc_4_postICA.m: Two files in, one file out
% INFO.preproc4_MIR_fNameLoaded = []; % Leave empty when initializing this file
% INFO.preproc4_W_fNameLoaded = []; % Leave empty when initializing this file
% INFO.preproc4_MC_fNameSaved = []; % Leave empty when initializing this file
% INFO.preproc4_analyzer = []; % Leave empty when initializing this file
% INFO.preproc4_datetime = []; % Leave empty when initializing this file




%%%%%%%%%%%%%%%%%%%%%%%%% Trigger info %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Triggers of interest once parsed from "LNNN" format of TCP triggers
% INFO.trigger.stim = [101:148 201:248 301:324 401:424 501:512 701:702]; % Triggers corresponding to stimuili
% INFO.trigger.response = [1:9 11 12]; % Triggers corresponding to number/true/false keys



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% ICA params %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INFO.ICA.W = [];   % Leave empty when initializing this file
% INFO.ICA.HiThresh = 0.30;   % High EOG corr threshold for automatic reject
% INFO.ICA.LowThresh = 0.20;  % Low EOG corr threshold for manual inspection
% INFO.ICA.HiReject = []; % Leave empty when initializing this file
% INFO.ICA.LowReject = []; % Leave empty when initializing this file
% INFO.ICA.EkgSrc = []; % Leave empty when initializing this file
% INFO.ICA.RmSrc = []; % Leave empty when initializing this file

