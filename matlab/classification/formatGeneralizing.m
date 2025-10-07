% testClassificationm
% ----------------------------------
% Ethan - August 1, 2024
%
% Basic example script running RSA on a single participant on Sherlock

clear all; close all; clc

addpath(genpath('/home/groups/brucemc/Analysis'));
%addpath(genpath('/Users/ethanroy/Documents/Stanford/EdNeuro/matlabCode'))

% load single particpant
load '/scratch/users/ethanroy/eeg_groupitizing_data/ENI_032.mat'

X = classData.X.xClean;

% 1: dots 2: digits, 3: math symbol
labels3 = classData.labels3; 

% 1: subitizing 2: counting, 3: math symbol
labels_range = classData.labels_range; 

% values 1-7 match numerosity 100: math symbol
labels_numerosity = classData.labels_numerosity;

dots_idx = find(labels3 == 1);
digits_idx = find(labels3 == 2);
numeric_idx = find(labels3~=3);
operand_idx = numeric_idx(mod(numeric_idx, 5) ~= 0);
numeric_operand_idx = find(labels3 == 2 & mod((1:length(labels3))', 5) ~= 0);

X_dots = X(:,:,dots_idx);
X_digits = X(:,:,numeric_operand_idx);

% across all three participants we can discriminate dots, digits, and
% symbols
dot_labels = labels_numerosity(dots_idx);
digit_labels = labels_numerosity(numeric_operand_idx);
operand_labels = labels_range(operand_idx);
 

% this code comes from example at: https://github.com/berneezy3/MatClassRSA/blob/dev2/examples/example_v2_visualization_plotMatrix.m
rnd_seed = 3;
n_trials_to_avg = 1;

RSA_dots = MatClassRSA;
RSA_digits = MatClassRSA;


% Data preprocessing (noise normalization, shuffling, pseudo-averaging),
% where the random seed is set to rnd_seed
[X_shuf_dots, Y_shuf_dots,rndIdx] = RSA_digits.Preprocessing.shuffleData(X_dots, dot_labels,'rngType', rnd_seed);
[X_shufNorm_dots, sigma_inv] = RSA_digits.Preprocessing.noiseNormalization(X_shuf_dots, Y_shuf_dots);
[X_shufNormAvg_dots, Y_shufAvg_dots] = RSA_digits.Preprocessing.averageTrials(X_shufNorm_dots, Y_shuf_dots, n_trials_to_avg, 'rngType', rnd_seed);

dot_class = RSA_dots.Classification.trainPairs(X_shufNormAvg_dots, Y_shufAvg_dots);
preds = RSA_dots.Classification.predict(dot_class, X_digits, 'actualLabels',digit_labels); 


