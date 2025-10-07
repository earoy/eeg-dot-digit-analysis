import numpy as np
from os import path as op
from itertools import combinations, pairwise
from tqdm import tqdm


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, train_test_split,permutation_test_score
from xgboost import XGBClassifier, XGBRegressor


def run_pairwise_uni_classification_perm(dataset, labels='labels_numerosity', format='combined', n_perm=1000):
    '''Run pairwise classification with cross validation and permutation tests to 
       set up accuracy matrices for RSA and generate p-values. Trains on different
       epochs of EEG data depending on value of labels'''

    pairwise_results_perm = {}

    # Classify overall numerosity from both dots and digits
    if format=='combined':
        X = dataset['X'][(dataset['labels_numerosity']<=5)]
        y = dataset['labels_numerosity'][(dataset['labels_numerosity']<=5)]
        result_str = 'numerosity_combined'

    # classify numerosity from dots only
    elif format=='dots':
        X = dataset['X'][(dataset['labels3']==1)]
        y = dataset[labels][(dataset['labels3']==1)]
        result_str = 'correcnumerosity_dots'

    # classify numerosity from digits only
    elif format=='digits':
        X = dataset['X'][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]
        y = dataset[labels][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]
        result_str = 'correcnumerosity_digits'

    # classify true/false from proposed solution
    elif format=='correct':
        X = dataset['X'][dataset['solution_idx']]
        y = dataset['labels_correct']
        result_str = 'correct_s5'

    # classify true/false from equals sign
    else:
        X = dataset['X'][dataset['eq_idx']]
        y = dataset['labels_correct']
        result_str = 'correct_s4'

    # set up individual pair for labels
    label_pairs = [(int(i[0]),int(i[1])) for i in list(combinations(list(np.unique(y)), 2)) if i[0]!=i[1]]

    for pair in tqdm(label_pairs):

        pca = PCA(n_components=0.99, svd_solver='full')
        le = LabelEncoder() # for xgboost classifier

	# Define the pipeline
        lda_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('lda', LinearDiscriminantAnalysis())  # LDA classifier
        ])

        pair_str = f"{pair[0]}_{pair[1]}"
        pairwise_results_perm[pair_str] = {}

        # grab data associated with each label pair
        label_mask = (y==pair[0])|(y==pair[1])
        X_mask = X[label_mask]
        y_mask = y[label_mask]

        # pca on eeg data
#        X_pca = pca.fit_transform(X_mask)

        # Perform cross-validation and collect results
        score, perm_scores, pvalue = permutation_test_score(
            lda_pipeline, X_mask, y_mask, scoring="accuracy",
            n_permutations=n_perm, n_jobs=4
        )

        # store results
        pairwise_results_perm[pair_str]['score'] = score
        pairwise_results_perm[pair_str]['perm_scores'] = perm_scores
        pairwise_results_perm[pair_str]['pvalue'] = pvalue

    return pairwise_results_perm
