
% se_cleaning.m
% -------------------------------------------------------------------------
% By Amilcar Malave (2025-01-16)

%%%%%%%%%%%%%%%% Intructions%%%%%%%%%%%%%%%%%%%%
% - This script directly references the se_cleaning_config.m file. Make sure 
%   to correctly fill each entry and enable/disable the desired steps. 
% - The script structure can be viewed by clicking the "Go To" button in 
%   the "EDITOR" tab.
% - Proper truncation is key to avoid data artifacts. 
%   Inspect the truncation figures.
% - In config file "Truncate Continuous Data" section, make 

%%%%%%%%%%%%%%%%%% Outputs %%%%%%%%%%%%%%%%%%%%% 
% The script writes out a _cleaned.mat file containing the following:
% - xClean: cleaned data, shape [channels, samples, epochs]
% - epoch_1: corresponding time vector for xClean samples
% - fs_RS: resampled/downsampled frequency (same if fs_RS = fs_0)
% 
% - INFO: structure containing key metadata and processing details relevant to the pipeline. 
%   Main Output variables:
%   - kept_Chs: list of channels kept after Removing "Tagged Bad Channels"
%     - bad_channels: list of channels removed
%     - onsets: onsets of kept epochs (legacy)
%     - epochs_trial_ids: (SSVEP Only) Which trial each epoch belongs to.
%     - categoryLabels: (ERP Only) Category/Condition for each epoch
%     - exemplarLabels: (OCED ERP Only) exemplarLabel for each epoch.

%%%%%%% Future Improvementes (some day) %%%%%%%%
% Fully Integrate single float precision for all steps (except filtering)
% Add optional extra conditions to filesearch
% Add optional name serach instead of auto file search
% add a word to not be included in the filename search
% Move more of the search into a function
% Optional TCPIP csv variable
% if some filter cutoff values are not entered, then skip that part (e.g.
    % low pass filter cutoff = [];
% text Journal
    % - The user can also choose to save out a journal, which details the
    %   following in a .txt file: Session information, input .mat filenames,
    %   filtering specifications, triggers and onsets.



fprintf('~ * ~ * Initiating Short Epoch Cleaning * ~ * ~\n\n')
tmpThisFile = "se_cleaning.m";

clear; close all; clc;

%% Specifications - Only need to specify items in this section once. 

% Specify config filename (user must ensure that config file is on the path)
% After creating your costume config file, replace the line below with yours

% Grabs config file from environment
configFn = getenv('config_path');
% configFn = '/home/users/ethanroy/eeg_groupitizing/code/bash/preproc/configs/se_cleaning_config_ENI_208.m';
addpath('/home/groups/brucemc/Analysis');

% Specify analyzer's initials
INFO.se_cleaning_analyzer = ''; % Specify analyzer's initials (move into config)

%% Main Preprocessing (Dont Touch)ls ~

% Load Config File
try  % Try config preliminary checks
    run(configFn) % Loading Config File
catch ME % If Preliminary Checks do not pass
    fprintf(2, '%s\n', ME.message); return % Print only the error message
end
INFO.configFn = configFn; clear configFn

% Load colors used in PLoS paper for easier interpretation of ERPs
load('colors.mat', 'rgb10')
config.rgb = rgb10; clear rgb10;

% Other variables
fs_0 = INFO.fs_0; % Original Sampling Freq (Hz)
fs_RS = INFO.FILTERING.Resample; % Resampling Freq (Hz)

% Getting the Onset Conditions for ERP
if INFO.study_style == "ERP"
    if tmp.onset_type{tmp.onset_type_id} == "DIN"
        INFO.onset_conditions = tmp.TCPIP_conditions;
    elseif tmp.onset_type{tmp.onset_type_id} == "Oced"
        INFO.onset_conditions = tmp.OCED_conditions;
    else
        error('CustomError:InvalidInput', 'Analysis not defined');
    end
end

clear tmp*


%% Searching Files to be Processed 

% Desired Files
INFO.file_labels.Matched_files = search_desired_files(INFO.dirs.raw, INFO.file_labels)'; 
% Number of files to be processed
nFilesMatRaw = length(INFO.file_labels.Matched_files); 
fprintf('\nSelected %d Files:\n', nFilesMatRaw); % Display number of files
disp(INFO.file_labels.Matched_files)
% Warning if no Files found - Stop Pipeline execution
if nFilesMatRaw == 0
    warning(['No Files Selected - Check "filename search and save strings" ' ...
        'and "INFO.dirs.raw" in config file' newline 'Cleaning Stopped']); return
end

% Output File Tag
if isscalar(INFO.file_labels.Subjects) % If only Onse subject -> outputs will contain subject number
    config.subStrOut = INFO.file_labels.Prefix + "_" + INFO.file_labels.Subjects{1};
else % If multiple subject -> Output name do not contain subject numbers
    config.subStrOut = INFO.file_labels.Prefix + "_" + INFO.file_labels.out_file;
end
fprintf('Output Tag: %s\n\n', config.subStrOut); % Outfile name tag


%% Load Data in Cell Arrays - (1 or more Subjects and/or Conditions)
fprintf('Loading Data\n')

[xIn,onsets,categoryLabels,exemplarLabels,sessionIDs] = loadData(INFO);

% Keep Onset data in different variable (corrected, uncorrected, lag)
if isstruct(onsets{1})
    Onsets_data = onsets;
    for i = 1:length(onsets)
        onsets{i} = onsets{i}.corrected;
    end
end


%% Truncate recordings (clean ends)

fprintf('\nTruncating\n');

% Truncating
[xTruncated, INFO] = truncateInput(xIn, INFO);

% Truncation Figures
if config.doFigs
    [figSt.truncStart, figSt.truncEnd] = plotTruncation(xIn, xTruncated, sessionIDs, INFO, fs_0, config);
end

% Fixing Onsets based on truncation
onsets_truncated = cell(1,nFilesMatRaw);
for i = 1:nFilesMatRaw
    onsets_truncated{i} = onsets{i} - INFO.truncateMsec.drift(i); % Account for start truncation
end


%% Filter Data 
fprintf('\nFiltering START'); vars.stime = tic;

% Filter info
% - All filters use filtfilt - zero-phase 
% - Notch: 8th-order Butterworth [2nd order x 2 (notch) x 2 (filtfilt) = 8th order]
% - Highpass: 8th-order Butterworth filter (4th order x 2 (filtfilt) = 8th order
% - Lowpass: 16th-order Chebyshev Type I filter. 8th order x 2 (filtfilt) = 16th order

xFiltered = cell(1,nFilesMatRaw); % Initialize Filtered Arraw 

for i = 1:nFilesMatRaw
    
    fprintf('\nRound %d of %d\n', [i,nFilesMatRaw]);

    if i == nFilesMatRaw
        % Fitering and Plotting Last Loaded Input
        [xFiltered{i}, INFO.FILTERING.filterSpecs, figSt.tH, figSt.fH] = ...
            doTrioFilterNoDS2(xTruncated{i}, INFO.fs_0, INFO.FILTERING.hpHz, ...
            INFO.FILTERING.notchHz, INFO.FILTERING.lpHz, config.doFilterFig, 'on');

        % Saving Filtering 
        if config.saveFigs && config.doFilterFig
            exportFig(figSt.tH,'_FilterTime_',INFO,config)
            exportFig(figSt.fH,'_FilterFreq_',INFO,config)
            if config.closeFigs; close all; end 
        end
        
    else
        % Filtering Inputs
        [xFiltered{i}, INFO.FILTERING.filterSpecs] = doTrioFilterNoDS2(...
            xTruncated{i}, INFO.fs_0, INFO.FILTERING.hpHz, ...
            INFO.FILTERING.notchHz, INFO.FILTERING.lpHz, 0, 'off');
    end
end

fprintf('Filtering END'); sectimer(vars.stime);
if config.freeVariables % Free RAM
    clear xTruncated xIn
end


%% Epoch data (Long epoch before Eye-Regression)

fprintf('\nEpoching (Long)\n');
% st = 1/INFO.fs_0*1000; % epoch step size
% epoch_0 = INFO.epoch.LStartMsec:st:INFO.epoch.LEndMsec-st;

[epoch_t0, epoch_s0, INFO] = getLongEpoch(INFO); 

xEpoched = cell(1,nFilesMatRaw); % Initialize xEpoched
INFO.epochs_per_file = ones(1,nFilesMatRaw);
onsets_all = cat(1, onsets_truncated{:}); % Concatenate all onsets
categoryLabels_all = cat(1, categoryLabels{:}); % Concatenate categorie labels
exemplarLabels_all = cat(1, exemplarLabels{:}); % Concatenate exemplar labels

vars.onsets_all = onsets_all;
vars.categoryLabels_all = categoryLabels_all;
vars.exemplarLabels_all = exemplarLabels_all;

switch INFO.study_style

    %%%%%%%% ERP Style %%%%%%%%%%%
    case "ERP"
    % Ensure the resampled data passes through zero
    % [epoch_fix, epoch_RS] = downsample_epoch_time(epoch_0,INFO);
    % Fixing epochs to match DS
    [epoch_t0, epochRS_t0, epoch_s0, epochRS_s0, INFO] = getEpochFix( ...
        epoch_t0, epoch_s0,INFO);
    % Epoching
    for i = 1:nFilesMatRaw
        xEpoched{i} = epochContinuousData(xFiltered{i}, ...
            onsets_truncated{i},epoch_s0,INFO);
        INFO.epochs_per_file(i) = size(xEpoched{i},3);
    end
    
    xEpoched = cat(3, xEpoched{:}); % All epochs in single variable
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%% SSVEP Style %%%%%%%%%%%
    case "SSVEP"
    % For now, it can only do 1 file at a time
    if nFilesMatRaw > 1
        error(['Epoch_ssvep can only take 1 file for now (cell array ' ...
            'length must = 1)', newline, 'Ensure xFiltered = 1x1 cell']);
    end
   
    % Epoch Identification Vector (Which trial they belong to)
    vars.epochsIDs = repelem(INFO.onsets.trials_keep, length(INFO.onsets.epochs_keep))';

    % Epoching SSVEP Data
    xEpoched_ssvep = cell(1,nFilesMatRaw); % Initialize xEpoched_ssvep
    for i = 1:nFilesMatRaw
        xEpoched_ssvep{i} = epoch_ssvep(xFiltered{i}, ...
            onsets_truncated{i},INFO);
    end
    xEpoched_ssvep = cat(3, xEpoched_ssvep{:}); % Concatenate Files
    
    [nElectrode, nSamples, ~, ~] = size(xEpoched_ssvep);
    xEpoched = reshape(xEpoched_ssvep,nElectrode, nSamples,[]);
    INFO.epochs_per_file(i) = size(xEpoched,3);

    clear xEpoched_ssvep;
    
    % Legacy comments
    % xEpoched_ssvep_flatted = reshape(xEpoched_ssvep, nelec, []);
    % xEpoched_ssvep_recovered = reshape(xEpoched_ssvep_flatted, ...
    %     nelec, nsamples, nepochs, ntrials);

    otherwise
        error('"INFO.study_style" not recognized');
end

[nElectrode, nSamples, nEpochs] = size(xEpoched);

if config.freeVariables % Free RAM
    clear xFiltered
end


%% Selected Channel for Plotting Figures

if isempty(config.ePlot) % If ePlot not defined in config file
    switch nElectrode
        case 74, config.ePlot = 58; % Easycap
        case 129, config.ePlot = 96; % EGI 129
        otherwise, error('Number of electrodes not recognized!');
    end
end
fprintf('\nPlots will use electrode number %s \n', num2str(config.ePlot))


%% Visualize Filtered Epoched data (concat epoch and ERP)

if config.doFigs
    % Concatenated raw trials (all electrodes)
    figSt.concat1 = plotConcatEpochs(xEpoched, INFO, config, ...
        '_Concat1_Raw_');
    
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP1 = plotERP(xEpoched, epoch_t0, ...
                categoryLabels_all, INFO, config, '_ERP1_Raw');
        case "SSVEP"
            figSt.FFT1 = plotFFT(xEpoched, fs_0, ...
                vars.epochsIDs, INFO, config, '_FFT1_Raw_');
    end
end


%% Eye Regression and Reduce Channels if using EGI

%%% EOG: Compute Eye Channels for Eye Regression
EOG.eog = computeEOG(xEpoched, EOG, nElectrode);

%%% Reduce channel montage if using EGI data
if nElectrode == 129
    % Keep Electrodes in config.useElectrodes (e.g. 10:124) 
    fprintf('\nReducing Channels\n');
    xEpoched = xEpoched(config.useElectrodes, :, :); 
end

%%%%%%%%%%%%%%%%%%%%% Regress out eye movements %%%%%%%%%%%%%%%%%%%%%%
if config.do_eye_regression % if = 1, do Eye Regression
    fprintf('\nPerforming Eye Regression\n');
    xRegress = nan(size(xEpoched)); % Space x time x trial
    
    % Regress out eye movements on a trial-by-trial basis. 
    % Long Epochs trials to keep more eye-related activity.
    for i = 1:nEpochs
        tempEEG = squeeze(xEpoched(:, :, i)); % space x time
        tempEOG = squeeze(EOG.eog(:, :, i)); % space x time
        xRegress(:,:,i) = regressOut(tempEEG, tempEOG);
        clear temp*
    end
    
    %%% Visualize Eye Regression - Before and After Regression
    if config.doFigs
        figSt.regression = plotEyeRegression(xEpoched, EOG.eog, xRegress, ...
            INFO, config, '_EOG_');
    end
    
    %%% Visualize Eye Regressed Data
    if config.doFigs
    
        % Concatenated trials (all electrodes)
        figSt.concat2 = plotConcatEpochs(xRegress, INFO, config, ...
            '_Concat2_EyeRegression_');
        
        % ERP (Continuous) or FFT (SSVEP)
        switch INFO.study_style
            case "ERP"
                figSt.ERP2 = plotERP(xRegress, epoch_t0, ...
                    categoryLabels_all, INFO, config, '_ERP2_EyeRegression_');
            case "SSVEP"
                figSt.FFT2 = plotFFT(xRegress, fs_0, ...
                    vars.epochsIDs, INFO, config, '_FFT2_EyeRegression_');
        end
    end

else % Skip "Eye Regression"
    fprintf('\nEye Regression Skiped\n');
    xRegress = xEpoched;
end

if config.freeVariables % Free RAM
    clear xEpoched
end



%% Bad Channels Fix

if config.fixBadChs % if = 1, do "Bad Channel Fix"
    fprintf('\nBad Channel Fix:\n');
    [xGoodChs, INFO] = fixBadChannels(xRegress, INFO, config, ...
        '_BadChannels_');
else % Skip "Bad Channel Fix"
    xGoodChs = xRegress;
end

if config.freeVariables % Free RAM
    clear xRegress
end

%%% Visualize After Bad Channels Fix 
if config.doFigs

    % Concatenated trials (all electrodes)
    figSt.concat3 = plotConcatEpochs(xGoodChs, INFO, config, ...
        '_Concat3_GoodChs_');
    
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP3 = plotERP(xGoodChs, epoch_t0, ...
                categoryLabels_all, INFO, config, '_ERP3_GoodChs_');
        case "SSVEP"
            figSt.FFT3= plotFFT(xGoodChs, fs_0, ...
                vars.epochsIDs, INFO, config, '_FFT3_GoodChs_');
    end
end


% Legacy
% %% Remove Tagged (Bad) Channels after Eye Regression
% 
% % Removed Bad Channels saved on INFO.bad_channels
% if config.fixBadChs && config.RemoveTaggedChs % if = 1 -> Remove Tagged Chs
%     fprintf('\nRemoving Tagged Bad Channels\n');
%     INFO.kept_Chs = setdiff(config.useElectrodes, INFO.bad_channels);
%     xImputed = xImputed(INFO.kept_Chs,:); % Keep only good channels
% 
%     % Change ePlot based on Bad channel Removal
%     config.ePlot = find(INFO.kept_Chs == config.ePlot);
%     fprintf('\nePlot = Ch %d\n', config.ePlot);
% end


%% Resample Epoched Data
fprintf('\nResampling\n');

switch INFO.study_style

    case "ERP"
        % Downsampling - No downsample if DS = 1 (xEpoched_RS = xEpoched)
        xEpoched_RS = zeros(size(xGoodChs,1),length(epochRS_t0),nEpochs); % Initialize Matrix
        for i = 1:size(xGoodChs,3)
            xEpoched_RS(:,:,i) = downsample(xGoodChs(:,:,i)',INFO.FILTERING.DS)';
        end

        % % % Resampling - No downsample if DS = 1 (xEpoched_RS = xEpoched) (fix)
        % temp_perm = permute(xEpoched, [2, 1, 3]);  % [Samples x Channel x Epoch]
        % temp_resample = resample(temp_perm, fs_RS, fs_0);
        % xEpoched_RS = permute(temp_resample, [2, 1, 3]);  % [Channel x Samples x Epoch]  

    case "SSVEP"
        st = 1/fs_RS*1000; % epoch step size
        epochRS_t0 = epoch_t0(1):st:epoch_t0(end); % Resampling Epoch
        % epochRS_s0 = epochRS_t0/st;

        % Resampling
        temp_perm = permute(xGoodChs, [2, 1, 3]);  % [Samples x Channel x Epoch]
        temp_resample = resample(temp_perm, fs_RS, fs_0); % Resampling
        xEpoched_RS = permute(temp_resample, [2, 1, 3]);  % [Channel x Samples x Epoch]
end

if config.freeVariables % Free RAM
    clear xGoodChs temp*
end

%%% Visualize Resampled Epoched data
if config.doFigs

    % Concatenated trials (all electrodes)
    figSt.concat4 = plotConcatEpochs(xEpoched_RS, INFO, config, ...
        '_Concat4_Resampled_');
    
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP4 = plotERP(xEpoched_RS, epochRS_t0, ...
                categoryLabels_all, INFO, config, '_ERP4_Resampled');
        case "SSVEP"
            figSt.FFT4 = plotFFT(xEpoched_RS, fs_RS, ...
                vars.epochsIDs, INFO, config, '_FFT4_Resampled_');
    end
end


%% Short Epochs (Epoch again)

% Trimming long epoch to short epoch
fprintf('\nRe-Epoching (small epochs)\n');
[xTrimmed, epoch_1] = epochShort(xEpoched_RS, epochRS_t0, fs_RS, INFO);

%%% Visualize shorter epochs
if config.doFigs

    % Concatenated trials (all electrodes)
    figSt.concat5 = plotConcatEpochs(xTrimmed, INFO, config, ...
        '_Concat5_ShortEpoch_');
    
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP5 = plotERP(xTrimmed, epoch_1, ...
                categoryLabels_all, INFO, config, '_ERP5_ShortEpoch_');
        case "SSVEP"
            figSt.FFT5= plotFFT(xTrimmed, fs_RS, ...
                vars.epochsIDs, INFO, config, '_FFT5_ShortEpoch_');
    end
end

if config.freeVariables % Free RAM
    clear xRegress
end


%% Exclude Bad Epochs and/or Fix Bad Samples and DC correct

% Preparing Matrix to NaN bad Samples
thisX_dc = dcCorrectMedian(xTrimmed); % Remove DC offset (Median) each channel
thisX_nan = cube2chRows(thisX_dc); % Concat trials to get NaNs

% NaN bad Samples
% NaN samples by standard deviation. Default = 4 iterations, 4 std.
if config.exclude.do_nan_std
    fprintf('\nNaNing Bad Samples based on std');
    % NaN bad samples based on standard deviation (default = 4 stds)
    thisX_nan = nanBadSamples(thisX_nan, 4, 4, 0, 25); % explain better
end
% NaN samples where voltage > Thresshold
if config.exclude.do_nan_voltageThres
    fprintf('\nNaNing Samples > %d μV', INFO.Thresh.recUV);
    this_nan_ids = abs(thisX_nan) > INFO.Thresh.recUV;
    thisX_nan(this_nan_ids) = NaN;
    disp("")
end

% Find NaN indices
that_nan_ids = (isnan(thisX_nan(:)));
thisIndicatorMatrix2D = reshape(that_nan_ids,size(thisX_nan,1),[]);

% Convert back to Epochs (3D)
thisX_nan3D = chRows2cube(thisX_nan, size(xTrimmed, 2));
thisIndicatorMatrix3D = chRows2cube(thisIndicatorMatrix2D, size(xTrimmed, 2));

% Excluding Bad Epochs
if config.exclude_bad_epochs % if = 1, then Exclude Bad Epochs
    fprintf('\nExcluding Bad Epochs\n');
    [xIncluded, vars.flags] = excludeBadEpochs2(thisX_nan3D, ...
        thisIndicatorMatrix3D , INFO.Thresh, 0);
    INFO.good_epochs = vars.flags.keep;
else % Skip "Exclude Bad Epochs"
    xIncluded = thisX_nan3D;
    vars.flags.keep = 1:size(xIncluded,3);
    INFO.good_epochs = vars.flags.keep;
end

% Labeling Bad Epochs
if config.label_bad_epochs % if = 1, then Exclude Bad Epochs
    fprintf('\nLabeling Bad Epochs\n');
    [ignore_this, vars.flags] = excludeBadEpochs2(thisX_nan3D, ...
        thisIndicatorMatrix3D , INFO.Thresh, 0);
    INFO.good_epochs = vars.flags.keep;
end

%%% Plot NaNs in data
if config.doFigs 
    figSt.NaNs = plotNaNs(thisX_dc,thisIndicatorMatrix2D, INFO, ...
        config, "_Bad_Samples_");     
end

%%% Update Segmentation Vectors Post Bad Epoch Exclusion
if ~config.excluded % if = 0, Vector segments have not been exluded yet
    INFO.onsets = onsets_all(vars.flags.keep);
   
    if INFO.study_style == "SSVEP"
        INFO.epochs_trial_ids = vars.epochsIDs(vars.flags.keep);
    
    elseif INFO.study_style == "ERP"
        INFO.categoryLabels = categoryLabels_all(vars.flags.keep);
        INFO.exemplarLabels = safeIndex(exemplarLabels_all,vars.flags.keep);
    end
    config.excluded = 1; % Vector segments excluded
end

if config.freeVariables % Free RAM
    clear xTrimmed
end
clear this* that*


%% Impute Missing Values

% DC correct trials
xIncluded_dc = dcCorrectMedian(xIncluded); % Correct using Median
this2D = cube2chRows(xIncluded_dc); % Concat Epochs for Imputation and AR

% Impute missing values
if config.Impute % Begin Imputation (config.Impute = 1)
    fprintf('\nImputing Missing Values\n');
    xImputed = imputeAllNaN74_128(this2D); % Imputing 
    
        % Saving Imputed NaNs figure
        figSt.nancol = figure(100); figSt.nancol.Visible = config.showFigs;
        grid on; grid minor;
        setFigureSaveSize(16, 6)
        if config.saveFigs
            exportFig(figSt.nancol,"_NaNsPerTimeSample_",INFO,config)
            if config.closeFigs
                close(figSt.nancol)
            end
        end
else
    xImputed = this2D; % Skip (config.Impute = 0)
end

%%% Plot Good and Imputed Epochs (concat epoch and ERP)
if config.doFigs

    thisImputed3D = chRows2cube(xImputed, size(xIncluded, 2));

    % Concatenated trials (all electrodes)
    figSt.concat6 = plotConcatEpochs(thisImputed3D, INFO, config, ...
        '_Concat6_GoodEpochs_');
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP6 = plotERP(thisImputed3D, epoch_1, ...
                INFO.categoryLabels, INFO, config, '_ERP6_GoodEpochs_');
        case "SSVEP"
            figSt.FFT6 = plotFFT(thisImputed3D, fs_RS, ...
                INFO.epochs_trial_ids, INFO, config, '_FFT6_GoodEpochs_');
    end
end

if config.freeVariables % Free RAM
    clear this* xIncluded_dc 
end


%% Final Clean (Average Reference and Epoch DC Correct)
fprintf('\nFinal Clean\n');

% Convert to average reference
if config.doAverageReference
    fprintf('\tPerforming Average Referencing\n');
    xAR = doAR(xImputed); 
else % Skip Average Reference
    xAR = xImputed;
end

% Convert back to Epochs (3D)
thisXCl3D = chRows2cube(xAR, size(xIncluded, 2));

% Subtract the DC offset of Each Trial (individually)
fprintf('\tFinal DC Correct\n');
xClean = dcCorrectTrial(thisXCl3D);

%%% Plot final clean data (concat epoch and ERP)
if config.doFigs
    % Concatenated trials (all electrodes)
    figSt.concat7 = plotConcatEpochs(xClean, INFO, config, ...
        '_Concat7_Clean_');
    % ERP (Continuous) or FFT (SSVEP)
    switch INFO.study_style
        case "ERP"
            figSt.ERP7 = plotERP(xClean, epoch_1, ...
                INFO.categoryLabels, INFO, config, '_ERP7_Clean_');
        case "SSVEP"
            figSt.FFT7 = plotFFT(xClean, fs_RS, ...
                INFO.epochs_trial_ids, INFO, config, '_FFT7_Clean_');
    end
end

clear this*
if config.freeVariables % Free RAM
    clear xAR xImputed xIncluded
end


%% Save output data

INFO.fs_RS = fs_RS; % Saving new sampling freq into INFO struct
INFO.epoch_1 = epoch_1; % Saving short epoch time in INFO 
INFO.subStrOut = config.subStrOut; % Saving output str in INFO

% OUTPUT IN A .MAT FILE.
if config.saveOutput
    % cd(INFO.dirs.cleaned)
    fnOut = INFO.dirs.cleaned + "/" + config.subStrOut + "_cleaned.mat";
    fprintf('\nSaving Output\n%s\n',fnOut);
    save(fnOut,'INFO','fs_RS','epoch_1', 'xClean')
    fprintf('\n\tSize of output data: %s\n', mat2str(size(xClean)));
end

fprintf('\nDONE\n');


