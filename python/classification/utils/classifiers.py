import numpy as np

from itertools import combinations
from scipy.stats import randint, uniform
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier

def run_pairwise_crossformat_classification(dataset, label):
    '''Perform binary classification training on either dots
       or digits and then see how well the models generalize
       to the other format. Training can happen on different
       sets of EEG data depending on the value of label'''

    if label=='labels_numerosity':
        labels = dataset['labels3']
        num_labels = dataset['labels_numerosity']
        labels_range = dataset['labels_range']

        X = dataset['X']

        # grab data associated with each label pair
        dot_mask = (labels==1)
        digit_mask = ((labels==2)&(labels_range==1))

    if label=='labels_sum':

        labels = dataset['trial_format']
        num_labels = dataset['labels_sum']

        X = dataset['X_multi']

        # grab data associated with each label pair
        dot_mask = (labels==1)
        digit_mask = (labels==2)

    else:

        labels = dataset['trial_format']
        num_labels = dataset['labels_sum']
        X = dataset['X'][dataset['eq_idx']]

        dot_mask = (labels==1)
        digit_mask = (labels==2)

    le = LabelEncoder() # for xgboost classifier

    xgb_pipeline_digits = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('xgb', XGBClassifier(random_state=42))  # XGB classifier
    ])

    xgb_pipeline_dots = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('xgb', XGBClassifier(random_state=42))  # XGB classifier
    ])


    X_dots = X[dot_mask]
    y_dots = num_labels[dot_mask]
    label_pairs_dots = [(int(i[0]),int(i[1])) for i in list(combinations(list(np.unique(y_dots)), 2)) if i[0]!=i[1]]


    X_digits = X[digit_mask]
    y_digits = num_labels[digit_mask]
    label_pairs_digits = [(int(i[0]),int(i[1])) for i in list(combinations(list(np.unique(y_digits)), 2)) if i[0]!=i[1]]


    pairwise_results_train_dots = {}
    pairwise_results_train_digits = {}

    # pca on eeg data
    # X_dots_pca_dots = pca_train_dots.fit_transform(X_dots)
    # X_digits_pca_dots = pca_train_dots.transform(X_digits)

    # X_digits_pca_digits = pca_train_dots.fit_transform(X_digits)
    # X_dots_pca_digits = pca_train_dots.transform(X_dots)

    for pair in tqdm(label_pairs_dots):


        pair_str = f"{pair[0]}_{pair[1]}"

        pairwise_results_train_dots[pair_str] = {}
        pairwise_results_train_digits[pair_str] = {}
        pair_label_mask_dots = (y_dots==pair[0])|(y_dots==pair[1])
        pair_label_mask_digits = (y_digits==pair[0])|(y_digits==pair[1])

        y_dots_pair = le.fit_transform(y_dots[pair_label_mask_dots])
        X_dots_pair = X_dots[pair_label_mask_dots]

        y_digits_pair = le.fit_transform(y_digits[pair_label_mask_digits])
        X_digits_pair = X_digits[pair_label_mask_digits]


        # Train on Dots
        xgb_pipeline_dots.fit(X_dots_pair, y_dots_pair)

        pairwise_results_train_dots[pair_str]['train_score'] = xgb_pipeline_dots.score(X_dots_pair, y_dots_pair)
        pairwise_results_train_dots[pair_str]['test_score'] = xgb_pipeline_dots.score(X_digits_pair, y_digits_pair)
        pairwise_results_train_dots[pair_str]['train_preds'] = xgb_pipeline_dots.predict(X_dots_pair)
        pairwise_results_train_dots[pair_str]['test_preds'] = xgb_pipeline_dots.predict(X_digits_pair)

        # Train on Digits
        xgb_pipeline_digits.fit(X_digits_pair, y_digits_pair)

        pairwise_results_train_digits[pair_str]['train_score'] = xgb_pipeline_digits.score(X_digits_pair, y_digits_pair)
        pairwise_results_train_digits[pair_str]['test_score'] = xgb_pipeline_digits.score(X_dots_pair, y_dots_pair)
        pairwise_results_train_digits[pair_str]['train_preds'] = xgb_pipeline_digits.predict(X_digits_pair)
        pairwise_results_train_digits[pair_str]['test_preds'] = xgb_pipeline_digits.predict(X_dots_pair)



    return pairwise_results_train_dots, pairwise_results_train_digits

def run_pairwise_uni_classification_cv(dataset, labels,format):
    '''Run pairwise classification with cross validation to 
       set up accuracy matrices for RSA. Trains on different
       epochs of EEG data depending on value of labels'''

    pairwise_results = {}

    if format=='dots':

        X = dataset['X'][(dataset['labels3']==1)]
        y = dataset[labels][(dataset['labels3']==1)]

    else:
        X = dataset['X'][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]
        y = dataset[labels][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]

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

        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('xgb', XGBClassifier(random_state=42))  # XGB classifier
        ])

        # Define 5-fold cross-validation strategy
        folds = min(10, min(sum(y==pair[0]),(sum(y==pair[1]))))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)


        pair_str = f"{pair[0]}_{pair[1]}"
        pairwise_results[pair_str] = {}

        # grab data associated with each label pair
        label_mask = (y==pair[0])|(y==pair[1])
        X_mask = X[label_mask]
        y_mask = le.fit_transform(y[label_mask])

        # pca on eeg data
        X_pca = pca.fit_transform(X_mask)

        # Perform cross-validation and collect results
        # lda_results = cross_validate(lda_pipeline, X_pca, y_mask, cv=cv, scoring=['accuracy'], return_estimator=True)
        xgb_results = cross_validate(xgb_pipeline, X_pca, y_mask, cv=cv, scoring=['accuracy'], return_estimator=True)

        # pairwise_results[pair_str]['lda_accuracies'] = lda_results['test_accuracy']
        # pairwise_results[pair_str]['lda_mean_accuracy'] = lda_results['test_accuracy'].mean()

        pairwise_results[pair_str]['xgb_accuracies'] = xgb_results['test_accuracy']
        pairwise_results[pair_str]['xgb_mean_accuracy'] = xgb_results['test_accuracy'].mean()

    return pairwise_results

def run_pairwise_classification_cv_tune(dataset, labels, train_format='both'):
    """
    Run pairwise classification with cross-validation to set up accuracy matrices for RSA.
    Now includes a grid search to tune XGBoost hyperparameters.
    """
    pairwise_results = {}

    # Select X and y depending on labels type
    if labels == 'labels_sum':
        X = dataset['X_multi']
        y = dataset[labels]
    elif labels == 'eq_sum':
        X = dataset['X'][dataset['eq_idx']]
        y = dataset['labels_sum']
    else:
        X = dataset['X']
        y = dataset[labels]

    if train_format!='both':
    
        if train_format=='dots':

            X = dataset['X'][(dataset['labels3']==1)]
            y = dataset[labels][(dataset['labels3']==1)]

        else:
            X = dataset['X'][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]
            y = dataset[labels][(dataset['labels3']==2)&(dataset['labels_numerosity']<=5)]


    # Create all unique label pairs
    label_pairs = [
        (int(i[0]), int(i[1]))
        for i in list(combinations(list(np.unique(y)), 2))
        if i[0] != i[1]
    ]

    # Set up 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Define a broader parameter distribution for XGBoost
    xgb_param_dist = {
        'xgb__n_estimators': randint(50, 300),
        'xgb__max_depth': randint(3, 10),
        'xgb__learning_rate': uniform(0.01, 0.3),
        'xgb__subsample': uniform(0.6, 0.4),
        'xgb__colsample_bytree': uniform(0.6, 0.4),
        'xgb__gamma': uniform(0, 5)
    }

    for pair in tqdm(label_pairs):
        pca = PCA(n_components=0.99, svd_solver='full')
        le = LabelEncoder()


        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
        ])

        pair_str = f"{pair[0]}_{pair[1]}"
        pairwise_results[pair_str] = {}

        # Get data for current label pair
        label_mask = (y == pair[0]) | (y == pair[1])
        X_mask = X[label_mask]
        y_mask = le.fit_transform(y[label_mask])

        # PCA dimensionality reduction
        X_pca = pca.fit_transform(X_mask)


        # Randomized search for XGBoost hyperparameters
        xgb_rand_search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions=xgb_param_dist,
            n_iter=20,             # number of random samples to try
            cv=cv,
            scoring='accuracy',
            n_jobs=4,
            verbose=0,
            random_state=42
        )
        xgb_rand_search.fit(X_pca, y_mask)

        # Get the best model and cross-validated scores
        best_xgb = xgb_rand_search.best_estimator_
        best_xgb_results = cross_validate(
            best_xgb,
            X_pca,
            y_mask,
            cv=cv,
            scoring=['accuracy']
        )


        pairwise_results[pair_str]['xgb_best_params'] = xgb_rand_search.best_params_
        pairwise_results[pair_str]['xgb_accuracies'] = best_xgb_results['test_accuracy']
        pairwise_results[pair_str]['xgb_mean_accuracy'] = best_xgb_results['test_accuracy'].mean()

    return pairwise_results

def run_pairwise_classification_cv(dataset, labels):
    '''Run pairwise classification with cross validation to 
       set up accuracy matrices for RSA. Trains on different
       epochs of EEG data depending on value of labels'''

    pairwise_results = {}

    if labels=='labels_sum':
        X = dataset['X_multi']
        y = dataset[labels]

    elif labels=='eq_sum':

        X = dataset['X'][dataset['eq_idx']]
        y = dataset['labels_sum']

    else:
        X = dataset['X']
        y = dataset[labels]

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

        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('xgb', XGBClassifier(random_state=42))  # XGB classifier
        ])

        # Define 5-fold cross-validation strategy
        folds = min(10, min(sum(y==pair[0]),(sum(y==pair[1]))))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)


        pair_str = f"{pair[0]}_{pair[1]}"
        pairwise_results[pair_str] = {}

        # grab data associated with each label pair
        label_mask = (y==pair[0])|(y==pair[1])
        X_mask = X[label_mask]
        y_mask = le.fit_transform(y[label_mask])

        # pca on eeg data
        X_pca = pca.fit_transform(X_mask)

        # Perform cross-validation and collect results
        lda_results = cross_validate(lda_pipeline, X_pca, y_mask, cv=cv, scoring=['accuracy'], return_estimator=True)
        xgb_results = cross_validate(xgb_pipeline, X_pca, y_mask, cv=cv, scoring=['accuracy'], return_estimator=True)

        pairwise_results[pair_str]['lda_accuracies'] = lda_results['test_accuracy']
        pairwise_results[pair_str]['lda_mean_accuracy'] = lda_results['test_accuracy'].mean()

        pairwise_results[pair_str]['xgb_accuracies'] = xgb_results['test_accuracy']
        pairwise_results[pair_str]['xgb_mean_accuracy'] = xgb_results['test_accuracy'].mean()

    return pairwise_results


def classify_epoch_correct(dataset, epoch='solution', form='combo'):
    '''Perform binary classification of whether a trial
       is true for false based on EEG data from a given epoch 
       (solution vs. =)'''
    
    
    X = dataset['X'][dataset[f'{epoch}_idx']]
    y = dataset['labels_correct']

    if form!='combo':

        if form=='dots':
            form_mask = dataset['labels3'][0::5]==1
        else:
            form_mask = dataset['labels3'][0::5]==2
       

        X = X[form_mask]
        y = y[form_mask]

    le = LabelEncoder() # for xgboost classifier
    y = le.fit_transform(y)

    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=0.99)),
        ('xgb', XGBClassifier(random_state=42))  # XGB classifier
    ])

    lda_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
#             ('pca', PCA(n_components=0.99)),
            ('lda', LinearDiscriminantAnalysis())  # LDA classifier
        ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_results = cross_validate(xgb_pipeline, X, y, cv=cv, scoring=['accuracy'], return_estimator=True)
    lda_results = cross_validate(lda_pipeline, X, y, cv=cv, scoring=['accuracy'], return_estimator=True)

    xgb_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.99)),
    ('xgb', XGBClassifier(random_state=42))
        ])
    
    xgb_estimator.fit(X, y)

    lda_estimator = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
#             ('pca', PCA(n_components=0.99)),
            ('lda', LinearDiscriminantAnalysis())  # LDA classifier
        ])
    lda_estimator.fit(X, y)

    return {'xgb':xgb_results, 'lda': lda_results,
            'xgb_estimator': xgb_estimator, 'lda_estimator': lda_estimator}


def timewise_classification(dataset, stim):
    '''Function to take an EEG dataset and run classification at
       each timepoint within an epoch'''

    # set up structures for storing results
    dataset[f'timepoint_modeling_{stim}'] = {}
    timewise_accuracies = np.array([])

    if stim=='labels3':
        # Set up Feature/Target structures for classification
        X = dataset['X'][dataset['labels_range']==1]
        y = dataset['labels3'][dataset['labels_range']==1]

    elif stim=='solution':
        X = dataset['X'][dataset['solution_idx']]
        y = dataset['labels_correct']

    else:
        X = dataset['X'][dataset['eq_idx']]
        y = dataset['labels_correct']


    n_samples = dataset['n_samples']

    for i in tqdm(range(0,n_samples+1)):

        time_idx = np.arange(i, X.shape[1], n_samples) # gives us each sensor at time point i

        le = LabelEncoder() # for xgboost classifier

        y = le.fit_transform(y)
        X_sub = X[:,time_idx]

        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('xgb', XGBClassifier(random_state=42))  # XGB classifier
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        xgb_results = cross_validate(xgb_pipeline, X_sub, y, cv=cv, scoring=['accuracy'], return_estimator=True)

        dataset[f'timepoint_modeling_{stim}'][str(i)] = xgb_results

        # add to list of average model performance across CV folds at each time point
        timewise_accuracies = np.append(timewise_accuracies, np.mean(xgb_results['test_accuracy']))


    dataset[f'timepoint_acc_{stim}'] = timewise_accuracies

    return dataset


def classify_epoch_correct_uni(dataset, epoch='solution', train_form='dots'):
    """
    Train classifiers to predict trial correctness (correct/incorrect) on one stimulus format 
    (dots or digits) and evaluate how well the model generalizes to the other format.

    This function supports evaluating classifier performance at two possible time points 
    during the trial: the presentation of the equals sign or the presentation of the solution, 
    controlled via the `epoch` parameter.

    Two classifiers are trained and evaluated:
        1. Linear Discriminant Analysis (LDA)
        2. Extreme Gradient Boosting (XGBoost)

    The results are stored in the input `dataset` dictionary under a key that encodes the 
    epoch and training format used.

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
    train_form : {'dots', 'digits'}, optional, default='dots'
        Which stimulus format to train on:
            - 'dots' → train on dot trials, test on digit trials.
            - 'digits' → train on digit trials, test on dot trials.

    Returns
    -------
    dataset : dict
        The input dataset dictionary with an additional key:
        `f'classify_correct_{epoch}_train_{train_form}_scores'`, 
        which stores a dictionary containing:
            - 'lda' : float
                Accuracy of the LDA classifier on the test set.
            - 'xgb' : float
                Accuracy of the XGBoost classifier on the test set.

    Notes
    -----
    - The function uses scikit-learn's `Pipeline` for standardization and classification.
    - The XGBoost classifier uses a fixed `random_state=42` for reproducibility.
    - The function modifies the input dataset in place.

    Examples
    --------
    >>> updated_dataset = classify_epoch_correct_uni(
    ...     dataset=my_dataset,
    ...     epoch='solution',
    ...     train_form='dots'
    ... )
    >>> updated_dataset['classify_correct_solution_train_dots_scores']
    {'lda': 0.78, 'xgb': 0.85}
    """

    dot_mask = dataset['labels3'][0::5]==1
    digit_mask = dataset['labels3'][0::5]==2

    dot_labels = dataset['labels_correct'][dot_mask]
    digit_labels = dataset['labels_correct'][digit_mask]

    if epoch=='solution':
        X_dots = dataset['X'][(dataset['solution_idx'])][dot_mask]
        X_digits = dataset['X'][(dataset['solution_idx'])][digit_mask]
    else:
        X_dots = dataset['X'][(dataset['eq_idx'])][dot_mask]
        X_digits = dataset['X'][(dataset['eq_idx'])][digit_mask]

    
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

    # Define the pipeline
    lda_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=0.99)),
        ('lda', LinearDiscriminantAnalysis())  # LDA classifier
    ])

    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=0.99)),
        ('xgb', XGBClassifier(random_state=42))  # XGB classifier
    ])

    # print(X_train.shape, y_train.shape)

    lda_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)

    score_lda = lda_pipeline.score(X_test,y_test)
    score_xgb = xgb_pipeline.score(X_test,y_test)

    dataset[f'classify_correct_{epoch}_train_{train_form}_scores'] = {'lda': score_lda, 'lda_estimator': lda_pipeline}
    dataset[f'classify_correct_{epoch}_train_{train_form}_scores']['xgb'] = score_xgb
    dataset[f'classify_correct_{epoch}_train_{train_form}_scores']['xgb_estimator'] = xgb_pipeline

    return dataset
