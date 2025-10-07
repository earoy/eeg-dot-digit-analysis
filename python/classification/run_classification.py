import pickle as pkl
import sys

from classifiers import *
from os import path as op
from utils import *
from sklearn.model_selection import permutation_test_score


if __name__== "__main__":
    
    if len(sys.argv) != 4:
        # no argumemnts, print usage message
        print("Usage:")
        print(" $ python run_classification.py <subID> <perms> <out_path>")
        sys.exit(0)
    
    # grab CLI arguments
    subject = sys.argv[1]
    run_permutations = sys.argv[2]
    out_path = sys.argv[3]

    dataset = load_data(f'{subject}.mat')

    labels = dataset['trial_format']
    correct_labels = dataset['labels_correct']
    X = dataset['X'][dataset['solution_idx']]

    dot_mask = (labels==1)
    digit_mask = (labels==2)

    # pull out dot and digit datasets
    X_dots = X[dot_mask]
    y_dots = correct_labels[dot_mask]

    X_digits = X[digit_mask]
    y_digits = correct_labels[digit_mask]

    dataset['full_res'] = classify_epoch_correct(dataset, 'solution')
    dataset['full_res_null'] = classify_epoch_correct(dataset, 'eq')

    dataset['generalized_results_null'], dataset['train_digits_pipe'], dataset['train_dots_pipe'] = train_generalization(dataset,epoch='eq')
    dataset['generalized_results'], dataset['train_digits_pipe'], dataset['train_dots_pipe'] = train_generalization(dataset)

    if run_permutations=='true' or run_permutations=='True':
        score, permutation_scores, pvalue = permutation_test_score(dataset['train_digits_pipe'],
                                                                X_dots, y_dots, n_jobs=8,
                                                                verbose=3,n_permutations=1000)
        
        dataset['null_dist'] = permutation_scores

    outfile_name = f'{subject}_results.pickle'

    with open(op.join(out_path, outfile_name), 'wb') as handle:
        pkl.dump(dataset, handle, protocol=pkl.HIGHEST_PROTOCOL)
