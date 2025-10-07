import numpy as np
import pickle as pkl
import sys

from utils.classifiers import *
from os import path as op
from utils.utils import *
from sklearn.model_selection import permutation_test_score

import os
os.environ["OMP_NUM_THREADS"] = "1"   # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"   # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads

def permutation_test_generalizing(dataset, train_form, epoch, n_perms=1000, model='lda'):
    """
    Run permutation tests to determine statistical significance of classifier cross-format
    generalizations.  

    This function supports evaluating classifier significance at two possible time points 
    during the trial: the presentation of the equals sign or the presentation of the solution, 
    controlled via the `epoch` parameter.

    The results are stored in three return variables: the observed score, the permuted scores,
    and the p-value. 

    Parameters
    ----------
    dataset : dict
        A dictionary containing EEG or neural response data and associated labels. 
        Must include the following keys:
            - 'X' : np.ndarray
                Array of shape (n_samples, n_features) containing trial data.
            - 'labels3' : np.ndarray
                Labels indicating stimulus type (1 = dots, 2 = digits).
            - 'labels_correct' : np.ndarray
                Binary labels for trial correctness (1 = correct, 0 = incorrect).
            - 'solution_idx' : slice or array-like
                Indices of the solution presentation epoch.
            - 'eq_idx' : slice or array-like
                Indices of the equals-sign presentation epoch.
    epoch : {'solution', 'eq'}, optional, default='solution'
        Which time point to analyze:
            - 'solution' → use data from the solution presentation.
            - any other value → use data from the equals-sign presentation.
    n_perms : int, optional, default=1000
        Number of permutations to perform to generate the null distribution.
    model : str, optional, default='lda'
        Which fitted model should we run our permutation tests with. Should be 
        saved in dataset as dataset[f'classify_correct_{epoch}_train_{train_form}_scores'][f'{model}_estimator']

    Returns
    -------
    obs_score: float
        The observed generalization accuracy of our estimator across formats
    perm_accs: list
        The distribution of accuracies generated during each of our permutations
    p_value: float
        The proportion of scores in the permuation distribution that are more 
        extreme than our observed score.  
    """

    # get observed score and extimator from our dataset
    obs_score = dataset[f'classify_correct_{epoch}_train_{train_form}_scores'][model]
    estimator = dataset[f'classify_correct_{epoch}_train_{train_form}_scores'][f'{model}_estimator']

    # grab just numerosities <=5
    dot_mask = dataset['labels3'][0::5]==1
    digit_mask = dataset['labels3'][0::5]==2

    # get correct/incorrect labels for each format
    dot_labels = dataset['labels_correct'][dot_mask]
    digit_labels = dataset['labels_correct'][digit_mask]

    # set which epoch to analyze
    if epoch=='solution':
        X_dots = dataset['X'][(dataset['solution_idx'])][dot_mask]
        X_digits = dataset['X'][(dataset['solution_idx'])][digit_mask]
    else:
        X_dots = dataset['X'][(dataset['eq_idx'])][dot_mask]
        X_digits = dataset['X'][(dataset['eq_idx'])][digit_mask]

    # set up the training and testing formats 
    if train_form=='dots':
        X_train = X_dots
        y_train = dot_labels

        X_test = X_digits
        y_test = digit_labels

    else:
        X_train = X_digits
        y_train = digit_labels

        X_test = X_dots
        y_test = dot_labels

    # initialize null distribution
    perm_accs = np.zeros(n_perms)

    for i in tqdm(range(n_perms)):
        # Shuffle training labels
        y_train_perm = np.random.permutation(y_train)
        
        # Refit on permuted labels
        estimator.fit(X_train, y_train_perm)
        y_pred_perm = estimator.predict(X_test)
        
        # Store permuted accuracy
        perm_accs[i] = accuracy_score(y_test, y_pred_perm)

    # Compute p-value
    p_value = (np.sum(perm_accs >= obs_score) + 1) / (n_perms + 1)
    # print(f"P-value: {p_value:.4f}")

    return obs_score, perm_accs, p_value


if __name__== "__main__":

    if len(sys.argv) < 4:
        # no argumemnts, print usage message
        print("Usage:")
        print(" $ python3 perm_correct_classification.py <participant> <input_dir> <output_dir>")
        sys.exit(0)
    
    # grab CLI arguments
    sub = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Time and compute similarities calculation
    results_dict = {}

    forms = ['dots','digits']
    epochs = ['solution', 'eq']

    results_dict[sub] = {}

    with open(op.join(input_dir, f'{sub}_classification_data.pkl'), 'rb') as f:
        dataset = pkl.load(f)
    
    for epoch in epochs:

        print(f'Running Permutation tests on {epoch}')

        # Run Permutation Tests for overall correct classification
        X_overall = dataset['X'][dataset[f'{epoch}_idx']]
        y_overall = dataset['labels_correct']
        score, permutation_scores, pvalue = permutation_test_score(dataset[f'{epoch}_class']['lda_estimator'],
                                                                   X_overall, 
                                                                   y_overall, 
                                                                   n_permutations=1000, n_jobs=4, verbose=3, random_state=42
        )

        results_dict[sub][f'overall_accuracy_{epoch}'] = {'score': score,
                                                          'permutation_scores':permutation_scores,
                                                          'pvals': pvalue}

        for form in forms:

            print(f'Running Permutation tests on {form}')

            # grab trials for either dot or digit trials
            if form=='dots':
                form_mask = dataset['labels3'][0::5]==1
            else:
                form_mask = dataset['labels3'][0::5]==2

            y_single = dataset['labels_correct'][form_mask]
            X_single = dataset['X'][(dataset[f'{epoch}_idx'])][form_mask]

            # run within-format permutaiton test
            score, permutation_scores, pvalue = permutation_test_score( dataset[f'{epoch}_{form}_class']['lda_estimator'],
                # dataset[f'classify_correct_{epoch}_train_{form}_scores']['lda_estimator'],
                                                                       X_single, 
                                                                       y_single, 
                                                                       n_permutations=1000, n_jobs=4, verbose=3, random_state=42
                                                                   )

            results_dict[sub][f'train_{form}_accuracy_{epoch}'] = {'score': score,
                                                                   'permutation_scores':permutation_scores,
                                                                   'pvals': pvalue}                                        

            # run across-format generalizing permutation test
            score_gen, permutation_scores_gen, pvalue_gen =  permutation_test_generalizing(dataset, form, epoch, n_perms=1000)

            results_dict[sub][f'train_{form}_generalizing_{epoch}'] = {'score_gen': score_gen,
                                                                       'permutation_scores_gen': permutation_scores_gen,
                                                                       'pvalue_gen':pvalue_gen}


        print(f'Saving results for {sub} in {output_dir}')
        with open(op.join(output_dir,f'{sub}_classification_perm_results.pkl'), 'wb') as f:
            pkl.dump(results_dict, f)
