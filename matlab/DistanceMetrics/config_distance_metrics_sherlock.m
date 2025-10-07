% config_distance_metrics_ER.m
% -------------------------
% Created 8/27/2025 by Ethan
%
% Config file for SENSI-analytic-prototyping repo.
%
% Instructions for use:
% 1. Create your own copy named 'config_XY.m' where 'XY' are e.g., your
%   initials or name. 
% 2. Update all paths to those on your local or analysis machine.
% 3. Update other variables as needed
%
% For now git will track any versions of this file. In the future we may
% configure the repo to ignore all but the core version.

%% Participant Info

subjectIDs = {"ENI_032", "ENI_195", "BLC_1043_2"};
n_cats = 10;

%% DATA PATHS

% Data path for image classification data (Kaneshiro et al., 2015)
datapath.classification_input = '/scratch/users/ethanroy/eeg_groupitizing_data/classification_input/pilot_2/';
datapath.output_dir = '/scratch/users/ethanroy/eeg_groupitizing_data/results/cvDistResults/';

%% CODE PATHS

% Analytic prototyping repo
% codepath.analyticPrototyping = '/Volumes/LaPuffin/EdNeuroData/SENSI-analytic-prototpying';

% BKanMatEEG repo
codepath.bkan = '/home/groups/brucemc/Analysis/BKanMatEEGToolbox';

% SRC repo to use
codepath.src = '/home/groups/brucemc/Analysis/SRC-dev-2025';

% RCA repo to use
codepath.rca = '/home/groups/brucemc/Analysis/RCA-dev-2025';

% MatClassRSA repo
codepath.matClassRSA = '/home/groups/brucemc/Analysis/MatClassRSA';
