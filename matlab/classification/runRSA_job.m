% Script to run RSA based on function call specified in config file
% params to modify:
% 	- participant

start = tic; 
addpath(genpath('/home/groups/brucemc/Analysis'));
addpath(genpath('/home/users/ethanroy/eeg_groupitizing/code/matlab'));
outPath = '/scratch/users/ethanroy/eeg_groupitizing_data/results/';
runRSA_call = getenv('function_call');

eval(runRSA_call);

elapsed = toc(start)
elapsed
