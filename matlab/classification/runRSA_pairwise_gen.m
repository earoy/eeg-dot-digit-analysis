function RSA_results = runRSA_pairwise_gen(subjectID, trainFormat, dataPath, outPath)
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


    formatSpec = '%s_train_%s_results.mat';
    outFile = sprintf(formatSpec,subjectID,trainFormat);
    
    X = classData.X.xClean;
    
    % 1: dots 2: digits, 3: math symbol
    labels3 = classData.labels3; 

    % values 1-7 match numerosity 100: math symbol
    labels_numerosity = classData.labels_numerosity;
    
    dots_idx = find(labels3 == 1);
    
    % only grab operand digits 
    digits_idx = find(labels3 == 2 & mod((1:length(labels3))', 5) ~= 0);
 
 
    X_dots = X(:,:,dots_idx);
    X_digits = X(:,:,digits_idx);
    
    % across all three participants we can discriminate dots, digits, and
    % symbols
    dot_labels = labels_numerosity(dots_idx);
    digit_labels = labels_numerosity(digits_idx);
    

    if trainFormat=="dots"
        X_train = X_dots;
        labels_train = dot_labels;
        X_test = X_digits;
        labels_test = digit_labels;
    else
        X_train = X_digits;
        labels_train = digit_labels;
        X_test = X_dots;
        labels_test = dot_labels;
    end
    
    % this code comes from example at: https://github.com/berneezy3/MatClassRSA/blob/dev2/examples/example_v2_visualization_plotMatrix.m
    rnd_seed = 3;
    n_trials_to_avg = 1;
    
    RSA = MatClassRSA;    
    
    % Data preprocessing (noise normalization, shuffling, pseudo-averaging),
    % where the random seed is set to rnd_seed
    [X_shuf_train, Y_shuf_train,rndIdx] = RSA.Preprocessing.shuffleData(X_train, labels_train,'rngType', rnd_seed);
    [X_shufNorm_train, sigma_inv] = RSA.Preprocessing.noiseNormalization(X_shuf_train, Y_shuf_train);
    [X_shufNormAvg_train, Y_shufAvg_train] = RSA.Preprocessing.averageTrials(X_shufNorm_train, Y_shuf_train, n_trials_to_avg, 'rngType', rnd_seed);
    
    train_class = RSA.Classification.trainPairs(X_shufNormAvg_train, Y_shufAvg_train);
    RSA_results = RSA.Classification.predict(train_class, X_test, 'actualLabels',labels_test); 
         
    save(strcat(outPath,outFile),'RSA_results','-v7.3')

end
