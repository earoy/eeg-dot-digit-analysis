function RSA_results = runRSA_classify_answer(subjectID, dataPath, outPath)
%---------------------------------------------------------------------------
%     subjectID = ENI_032;
%     dataPath = '/Users/Ethan/Documents/Stanford/EdNeuro/project_eeg_classification_bootstrap/Data/preproc/S06.mat';
%---------------------------------------------------------------------------
%
%     Function to run RSA based on parameters
%     Input Args:
%       - subjectID: subjectID
%
%      Output Args:
%        - RSA_results: list of nBoot RSA result structs


    rnd_seed = 3;
    load(dataPath);


    formatSpec = '%s_classifyAnswer_results.mat';
    outFile = sprintf(formatSpec,subjectID);
    
    X = classData.X.xClean;
    
    % 1: dots 2: digits, 3: math symbol
    labels3 = classData.labels3; 

    % values 1-7 match numerosity 100: math symbol
    labels_numerosity = classData.labels_numerosity;
    
    dots_idx = find(labels3 == 1);
    digits_idx = find(labels3 == 2);
    numeric_idx = find(labels3~=3);
    operand_idx = numeric_idx(mod(numeric_idx, 5) ~= 0);

    X_operands = X(:,:,operand_idx);
    X_concat_operands = cat(2, X_operands(:,:,1:2:end), X_operands(:,:,2:2:end));

    % across all three participants we can discriminate dots, digits, and
    % symbols
    sum_labels = sum(reshape(labels_numerosity(operand_idx), 2, []), 1); 
   
    % this code comes from example at: https://github.com/berneezy3/MatClassRSA/blob/dev2/examples/example_v2_visualization_plotMatrix.m
    rnd_seed = 3;
    n_trials_to_avg = 1;
    
    RSA = MatClassRSA;    
    
    % Data preprocessing (noise normalization, shuffling, pseudo-averaging),
    % where the random seed is set to rnd_seed
    [X_shuf_train, Y_shuf_train,rndIdx] = RSA.Preprocessing.shuffleData(X_concat_operands, sum_labels,'rngType', rnd_seed);
    [X_shufNorm_train, sigma_inv] = RSA.Preprocessing.noiseNormalization(X_shuf_train, Y_shuf_train);
    [X_shufNormAvg_train, Y_shufAvg_train] = RSA.Preprocessing.averageTrials(X_shufNorm_train, Y_shuf_train, n_trials_to_avg, 'rngType', rnd_seed);
    
    RSA_results = RSA.Classification.crossValidatePairs(X_shufNormAvg_train, Y_shufAvg_train, 'nFolds', 3);
         
    save(strcat(outPath,outFile),'RSA_results','-v7.3')

end
