import mne
import numpy as np
import pickle as pkl
import sys
import os

from utils.classifiers import *
from os import path as op
from utils.utils import *
from utils.mne_utils import *
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator,get_coef, cross_val_multiscore, Vectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


os.environ["OMP_NUM_THREADS"] = "2"   # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "2"   # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # OpenBLAS threads

if __name__== "__main__":

    ## Run this script after running epoch_classification.py.py

    if len(sys.argv) < 4:
        # no argumemnts, print usage message
        print("Usage:")
        print(" $ python3 mne_classification.py <participant> <montage_path> <results_dir>")
        sys.exit(0)
    
    # grab CLI arguments
    participant = sys.argv[1]
    montage_path = sys.argv[2]
    results_dir = sys.argv[3]

    # create info
    n_channels = 124
    sampling_freq = 200  # in Hertz
    info = mne.create_info(sfreq=sampling_freq,
                        ch_names=[str(ch) for ch in list(range(1,n_channels+1))],
                        ch_types=['eeg']*124)

    info.set_montage(make_custom_montage(montage_path))

    # Time and compute similarities calculation
    with open(op.join(results_dir, f'{participant}_classification_data.pkl'), 'rb') as f:
        dataset = pkl.load(f)

    # set up data for classifier analysis
    # grab just the indices of the operands
    op_idx = np.concat([dataset['op_1_idx'], dataset['op_2_idx']])
    solution_idx = dataset['solution_idx']
    eq_idx = dataset['eq_idx']

    dot_idx = dataset['trial_format']==1
    digit_idx = dataset['trial_format']==2

    dot_idx_op = get_op_idx_by_format(dot_idx)
    digit_idx_op = get_op_idx_by_format(digit_idx)

    print('Making epochs...')
    num_epoch = make_epochs(dataset, info, dataset['labels_numerosity'][op_idx], 
                        dict(one=1, two=2, three=3, four=4, five=5), 
                        dataset['labels_numerosity'][op_idx].shape[0], op_idx)

    format_epoch = make_epochs(dataset, info, dataset['labels3'][op_idx], 
                           dict(dots=1, digits=2), 
                           dataset['labels3'][op_idx].shape[0], op_idx)

    solution_epoch = make_epochs(dataset, info, dataset['labels_correct'], 
                             dict(correct=1, incorrect=0),
                             dataset['labels_correct'].shape[0], solution_idx)

    eq_epoch = make_epochs(dataset, info, dataset['labels_correct'],
                       dict(correct=1, incorrect=0),
                       dataset['labels_correct'].shape[0], eq_idx)
    
    solution_epoch_dot = make_epochs(dataset, info, dataset['labels_correct'][dot_idx],
                       dict(correct=1, incorrect=0),
                       dataset['labels_correct'][dot_idx].shape[0], solution_idx[dot_idx])

    solution_epoch_digit = make_epochs(dataset, info, dataset['labels_correct'][digit_idx],
                       dict(correct=1, incorrect=0),
                       dataset['labels_correct'][digit_idx].shape[0], solution_idx[digit_idx])

    eq_epoch_dot = make_epochs(dataset, info, dataset['labels_correct'][dot_idx],
                       dict(correct=1, incorrect=0),
                       dataset['labels_correct'][dot_idx].shape[0], eq_idx[dot_idx]) 

    eq_epoch_digit = make_epochs(dataset, info, dataset['labels_correct'][digit_idx],
                   dict(correct=1, incorrect=0),
                   dataset['labels_correct'][digit_idx].shape[0], eq_idx[digit_idx]) 
    
    print('Running Within-Format LDA Classifications')
    dataset['mne_correct_results_sol'] = run_LDA(solution_epoch,['correct','incorrect'])
    dataset['mne_correct_results_eq'] = run_LDA(eq_epoch,['correct','incorrect'])
    dataset['mne_correct_results_sol_dot'] = run_LDA(solution_epoch_dot,['correct','incorrect'])
    dataset['mne_correct_results_sol_digit'] = run_LDA(solution_epoch_digit,['correct','incorrect'])
    dataset['mne_correct_results_eq_dot'] = run_LDA(eq_epoch_dot,['correct','incorrect'])
    dataset['mne_correct_results_eq_digit'] = run_LDA(eq_epoch_digit,['correct','incorrect'])
    
    print('Running Cross-Format LDA Generalization')
    dataset['mne_correct_results_sol_dot_train_gen'] = run_LDA_gen(solution_epoch_dot,
                                                                   solution_epoch_digit,['correct','incorrect'])
    
    dataset['mne_correct_results_sol_digit_train_gen'] = run_LDA_gen(solution_epoch_digit,
                                                                   solution_epoch_dot,['correct','incorrect'])

    
    print('Running Within-Format Temporal Generalization')
    gen_epochs = {'solution_digit':solution_epoch_digit, 
                  'solution_dot': solution_epoch_dot,
                  'solution':solution_epoch
                 }

    for epoch in gen_epochs.keys():

        X = gen_epochs[epoch].get_data(copy=False)
        y = gen_epochs[epoch].events[:, 2]

        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.99),
            #  LinearModel(LogisticRegression(solver="liblinear")),  # liblinear is faster than lbfgs
            LinearModel(LinearDiscriminantAnalysis(solver="lsqr"))
        )

        time_gen_corr = GeneralizingEstimator(clf, scoring="accuracy", n_jobs=4, verbose=True)
        time_gen_corr.fit(X=X , y=y)

        dataset[f'mne_{epoch}_temp_gen_results'] =  time_gen_corr.score(X=X, y=y)
        
    
    res_list = {}
    res_list_num_format_gen = {}

    # Train sliding estimator and temporal generalizer
    print('Running Time-resolved Classifications')
    for i in range(1,6):
        num_idx = dataset['labels_numerosity']==i
        n_trials = int(sum(num_idx))
        
        num_format_epochs = make_epochs(dataset, info, dataset['labels3'][num_idx],
                                                dict(dots=1, digits=2), n_trials,
                                                num_idx)

        print(f'{i}: dots vs digits')
        res_list[i] = run_LDA(num_format_epochs,['dots','digits'])

        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.99),
            #  LinearModel(LogisticRegression(solver="liblinear")),  # liblinear is faster than lbfgs
            LinearDiscriminantAnalysis(solver="lsqr")
        )
        time_gen_corr = GeneralizingEstimator(clf, scoring="accuracy", n_jobs=4, verbose=True)
        time_gen_corr.fit(X=num_format_epochs.get_data(copy=False) , y=num_format_epochs.events[:, 2])

        res_list_num_format_gen[i] = time_gen_corr.score(
            X=num_format_epochs.get_data(copy=False), y=num_format_epochs.events[:, 2]
        )

    dataset['mne_format_within_numerosity_results'] = res_list
    dataset['mne_format_within_numerosity_generalizing_results'] = res_list_num_format_gen

    # save the data
    print(f'Saving results for {participant} in {results_dir}')
    with open(op.join(results_dir,f'{participant}_classification_data.pkl'), 'wb') as f:
        pkl.dump(dataset, f)
