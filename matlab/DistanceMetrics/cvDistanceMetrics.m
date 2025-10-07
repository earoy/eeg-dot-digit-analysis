clear all; close all; clc

% Run config file; add to path as needed
run('config_distance_metrics_sherlock.m')
addpath(genpath(codepath.rca))
addpath(genpath(codepath.bkan))
addpath(genpath(codepath.matClassRSA))

% Check if the directory exists
if ~isfolder(fullfile(datapath.output_dir, 'Figs'))
    % If it doesn't exist, create it
    mkdir(fullfile(datapath.output_dir, 'Figs'));
end

 % Create the base sequence (1 to j)
base_sequence = 1:5;
% Calculate how many times to repeat the base sequence
num_repetitions = ceil(800 / 5);
% Repeat the base sequence
op_position = repmat(base_sequence, 1, num_repetitions);

resRDM_pear_fin = zeros(length(n_cats));
resRDM_euc_fin = zeros(length(n_cats));

for sub = 1:length(subjectIDs)
    
    subjectID = subjectIDs{sub};

    load(fullfile(datapath.classification_input,subjectID+".mat"));

    num_mask = classData.labels_numerosity < 6;
    symb_mask = classData.labels3 ~= 3;
    first_mask = op_position==1;
    third_mask = op_position==3;
    
    num_form = arrayfun(@(a,b) sprintf('%d_%d', a, b), classData.labels3, ...
        classData.labels_numerosity, 'UniformOutput', false);

    num_form = string(num_form);
    
    % Find unique values of numerosity and format and map them to sequential integers
    [num_form_labs, ~, num_form] = unique(num_form);
    
    % filter X and y so it just has data for numbers 1-5
    X = classData.X(:,:,(num_mask(:)));
    y = num_form(num_mask);
    
    % get the number of time points per epoch
    n_times = size(X);
    n_times = n_times(2);
    
    % calculate RDMs based on CV distance metrics
    resRDM_pear = zeros(length(unique(y)));
    resRDM_euc = zeros(length(unique(y)));
    
    for i=1:n_times
        X_sing = squeeze(X(:,i,:));
        D = RDM_Computation.computePearsonRDM(X_sing, y);
        resRDM_pear = resRDM_pear + D.RDM;
    
        D = RDM_Computation.computeEuclideanRDM(X_sing, y);
        resRDM_euc = resRDM_euc + D.RDM;
    end
    
    resRDM_pear = resRDM_pear/n_times;
    resRDM_euc = resRDM_euc/n_times;

    resRDM_pear_fin = resRDM_pear_fin + resRDM_pear;
    resRDM_euc_fin = resRDM_euc_fin + resRDM_euc;

    %% Save Individual Figures -- Pearson
    Visualization.plotMatrix(resRDM_pear);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_PearsonRDM_' thisDateTime(0) '.png'], [8 6])

    close all

    %% Save Individual Figures -- Euclidean Distance
    Visualization.plotMatrix(resRDM_euc);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_EucRDM_' thisDateTime(0) '.png'], [8 6])
    close all

        %% Save Individual Figures -- Pearson
    Visualization.plotDendrogram(resRDM_pear);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_PearsonDendrogram_' thisDateTime(0) '.png'], [8 6])

    close all

    %% Save Individual Figures -- Euclidean Distance
    Visualization.plotDendrogram(resRDM_euc);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_EucDenrogram_' thisDateTime(0) '.png'], [8 6])
    close all

        %% Save Individual Figures -- Pearson
    Visualization.plotMDS(resRDM_pear);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_PearsonMDS_' thisDateTime(0) '.png'], [8 6])

    close all

    %% Save Individual Figures -- Euclidean Distance
    Visualization.plotMDS(resRDM_euc);
    saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
        [sprintf('%s', subjectID) '_EucMDS_' thisDateTime(0) '.png'], [8 6])
    close all

end

resRDM_pear_fin = resRDM_pear_fin/length(subjectIDs);
resRDM_euc_fin = resRDM_euc_fin/length(subjectIDs);

save(fullfile(datapath.output_dir,['resRDM_pears_fin_' thisDateTime(0) '.mat']), "resRDM_pear_fin");
save(fullfile(datapath.output_dir,['resRDM_euc_fin_' thisDateTime(0) '.mat']), "resRDM_euc_fin");

%% Save Individual Figures -- Pearson
Visualization.plotMatrix(resRDM_pear_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_PearsonRDM_' thisDateTime(0) '.png'], [8 6])

close all

%% Save Individual Figures -- Euclidean Distance
Visualization.plotMatrix(resRDM_euc_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_EucRDM_' thisDateTime(0) '.png'], [8 6])
close all

%% Save Individual Figures -- Pearson
Visualization.plotDendrogram(resRDM_pear_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_PearsonDendrogram_' thisDateTime(0) '.png'], [8 6])

close all

%% Save Individual Figures -- Euclidean Distance
Visualization.plotDendrogram(resRDM_euc_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_EucDendrogram_' thisDateTime(0) '.png'], [8 6])
close all

%% Save Individual Figures -- Pearson
Visualization.plotMDS(resRDM_pear_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_PearsonMDS_' thisDateTime(0) '.png'], [8 6])

close all

%% Save Individual Figures -- Euclidean Distance
Visualization.plotMDS(resRDM_euc_fin);
saveCurrentFigure(fullfile(datapath.output_dir, 'Figs'), ...
    ['Full_EucMDS_' thisDateTime(0) '.png'], [8 6])
close all

% Visualization.plotMatrix(resRDM_pear);
% Visualization.plotMatrix(resRDM_euc);
