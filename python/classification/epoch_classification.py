import numpy as np
import pickle as pkl
import sys

from utils.classifiers import *
from os import path as op
from utils.utils import *

import os
os.environ["OMP_NUM_THREADS"] = "2"   # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "2"   # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # OpenBLAS threads

def run_epoch_classifications(participant, dir):
    """Run classification analysis on entire epoch of a given particpant.

    Parameters
    ----------
    participant : str
        The participantID to specify which participant to analyze.
    dir : str
        Input directory where data are located.

    Returns
    -------
    participant_result : dict
        A dictionary containing the EEG and trial level data, as well
        as the results from the various classifiers.
    """

    print(f'Loading data for {participant}')
    # Load data for pilot 1
    participant_result = load_data(f'{participant}.mat', dir)

    # classifiy operand position across both formats
    participant_result['labels_trial_idx'] = np.array([i%5 + 1 for i in range(participant_result['X'].shape[0])])
    participant_result['pairwise_stim_idx_results'] = run_pairwise_classification_cv(participant_result, 'labels_trial_idx')
    
    # classify numerosity overall across both formats
    participant_result['pairwise_results'] = run_pairwise_classification_cv(participant_result, 'labels_numerosity')
    
     # classify numerosity+format 
    participant_result['pairwise_results_num_form'] = run_pairwise_classification_cv(participant_result, 'labels_num_form')

    # classify answer from operands across both formates
    participant_result['pairwise_results_operands'] = run_pairwise_classification_cv(participant_result, 'labels_sum')
    
    # classify answer from equals sign across both formats
    participant_result['pairwise_results_sum_eq'] = run_pairwise_classification_cv(participant_result, 'eq_sum')

    # cross-modal predictions of operand numerosity training on either dots or digits 
    pairwise_results_train_dots, pairwise_results_train_digits = run_pairwise_crossformat_classification(participant_result,'labels_numerosity')

    participant_result['pairwise_results_train_dots'] = pairwise_results_train_dots
    participant_result['pairwise_results_train_digits'] = pairwise_results_train_digits

    # cross-modal predictions of solution training on operands
    pairwise_results_train_dots_sum, pairwise_results_train_digits_sum = run_pairwise_crossformat_classification(participant_result,'labels_sum')

    participant_result['pairwise_results_train_dots_sum'] = pairwise_results_train_dots_sum
    participant_result['pairwise_results_train_digits_sum'] = pairwise_results_train_digits_sum

    # cross-modal predictions of solution training on either dots or digits at '='
    pairwise_results_eq_train_dots, pairwise_results_eq_train_digits = run_pairwise_crossformat_classification(participant_result, 'eq_sum')
    participant_result['pairwise_results_eq_train_dots'] = pairwise_results_eq_train_dots
    participant_result['pairwise_results_eq_train_digits'] = pairwise_results_eq_train_digits

    # Train model on trials from one format to predict numerosity and test on other format
    participant_result['pairwise_uni_results_digits'] = run_pairwise_uni_classification_cv(participant_result,'labels_numerosity','digits')
    participant_result['pairwise_uni_results_dots'] = run_pairwise_uni_classification_cv(participant_result,'labels_numerosity','dots')

    # Perform RandomSearchCV hyperparameter tuning for num_form labels
    participant_result['num_form_tune'] = run_pairwise_classification_cv_tune(participant_result, 'labels_num_form')

    ## Train correct/incorrect classifier on both formats

    # Overall correct/incorrect classification
    participant_result['solution_class'] = classify_epoch_correct(participant_result, 'solution')
    participant_result['eq_class'] = classify_epoch_correct(participant_result, 'equals')
    
    ## Within-format correct/incorrect classification
    participant_result['solution_dots_class'] = classify_epoch_correct(participant_result, 'solution', 'dots')
    participant_result['eq_dots_class'] = classify_epoch_correct(participant_result, 'eq','dots')
    participant_result['solution_digits_class'] = classify_epoch_correct(participant_result, 'eq','symb')
    participant_result['eq_digits_class'] = classify_epoch_correct(participant_result, 'eq','symb')


    ## Train correct/incorrect classifier on single format and test generalization
    
    classify_epoch_correct_uni(participant_result,epoch='solution',train_form='digits')
    classify_epoch_correct_uni(participant_result,epoch='solution',train_form='dots')
    classify_epoch_correct_uni(participant_result,epoch='eq',train_form='digits')
    classify_epoch_correct_uni(participant_result,epoch='eq',train_form='dots')

    return participant_result

if __name__== "__main__":

    if len(sys.argv) < 4:
        # no argumemnts, print usage message
        print("Usage:")
        print(" $ python3 epoch_classification.py <participant> <input_dir> <output_dir>")
        sys.exit(0)
    
    # grab CLI arguments
    participant = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Time and compute similarities calculation
    result = run_epoch_classifications(participant, input_dir)

    print(f'Saving results for {participant} in {output_dir}')
    with open(op.join(output_dir,f'{participant}_classification_data.pkl'), 'wb') as f:
        pkl.dump(result, f)

