import pandas as pd
import numpy as np
import scipy as sp
import mne
from os import path as op
from itertools import combinations
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import stats as stats

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, GridSearchCV, train_test_split,permutation_test_score
from utils import *
from xgboost import XGBClassifier, XGBRegressor
from mne.stats import permutation_cluster_1samp_test


from mne.datasets import sample
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator,get_coef, cross_val_multiscore, Vectorizer
from mne.io.constants import FIFF

from scipy.stats import wilcoxon

def cross_classification(dataset, X, y):
    '''Function to take an EEG dataset and run classification at
       each timepoint within an epoch'''

    timepoints = []

    for i in tqdm(range(0,dataset['n_samples'])):

        train_time_idx = np.arange(i, X.shape[1], dataset['n_samples']) # gives us each sensor at time point i (for training)

        xgb_pipeline = Pipeline([
                    ('scaler', StandardScaler()),  # Standardize features
                    ('xgb', XGBClassifier(random_state=42))  # XGB classifier
                ])
        
        xgb_pipeline_cv = Pipeline([
                    ('scaler', StandardScaler()),  # Standardize features
                    ('xgb', XGBClassifier(random_state=42))  # XGB classifier
                ])
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        le = LabelEncoder() # for xgboost classifier

        X_train = X[:,train_time_idx]
        y = le.fit_transform(y)

        # fit model using training time point in the outer loop
        xgb_pipeline.fit(X_train, y)

        for j in range(0,dataset['n_samples']):
            
            # set up train/test data based on indices
            test_time_idx = np.arange(j, X.shape[1], dataset['n_samples']) # gives us each sensor at time point j (for testing)
            X_test = X[:,test_time_idx]

            # run 10-fold CV to train/test at the same time point    
            if i == j:

                xgb_results = cross_validate(xgb_pipeline_cv, X_train, y, cv=cv, scoring=['accuracy'], return_estimator=True)
                test_score = np.mean(xgb_results['test_accuracy'])
            
            # otherwise train on time i and test on time j
            else:
                # predict other time points from trained model
                test_score = xgb_pipeline.score(X_test,y)
                
                
            timepoints.append((i,j,test_score))

    return timepoints


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


def get_shap_values(estimator, X, background_samples=100, verbose=None):
    """Retrieve the SHAP values from an estimator ending with XGBoost.

    Parameters
    ----------
    estimator : object
        A fitted estimator from scikit-learn with XGBoost as final step.
    X : array-like
        Input data to compute SHAP values for.
    background_samples : int
        Number of background samples to use for SHAP explainer.
    verbose : bool
        Verbosity level.

    Returns
    -------
    shap_values : array
        The mean absolute SHAP values across samples for each feature and time point.
    """
    # Get the estimator
    est = estimator
    while hasattr(est, "steps"):
        est = est.steps[-1][1]

    squeeze_first_dim = False
    
    # If SlidingEstimator, loop across estimators
    if hasattr(est, "estimators_"):
        shap_values_list = []
        
        for ei, this_est in enumerate(est.estimators_):
            if ei == 0:
                print("  Extracting SHAP values from SlidingEstimator.")
            
            # Get the XGBoost model from the pipeline
            xgb_model = this_est.steps[-1][1]  # Get XGBoost from pipeline
            
            # Transform the data through the preprocessing steps
            X_transformed = X[:, :, ei]  # Get data for this time point
            for step_name, transformer in this_est.steps[:-1]:
                X_transformed = transformer.transform(X_transformed)
            
            # Create SHAP explainer
            # Use a subset of the data as background for efficiency
            background_indices = np.random.choice(
                X_transformed.shape[0], 
                size=min(background_samples, X_transformed.shape[0]), 
                replace=False
            )
            background = X_transformed[background_indices]
            
            explainer = shap.TreeExplainer(xgb_model, background)
            
            # Calculate SHAP values for all samples
            shap_vals = explainer.shap_values(X_transformed)
            
            # If binary classification, shap_vals might be 3D, take the positive class
            if len(shap_vals.shape) == 2:
                mean_shap = np.mean(np.abs(shap_vals), axis=0)
            else:
                # For binary classification, XGBoost sometimes returns values for positive class only
                mean_shap = np.mean(np.abs(shap_vals), axis=0)
            
            shap_values_list.append(mean_shap)
        
        shap_values = np.array(shap_values_list).T  # Transpose to match expected format
        shap_values = shap_values[np.newaxis]  # Add fake sample dimension
        squeeze_first_dim = True
    else:
        raise ValueError("This function is designed for SlidingEstimator with XGBoost")

    if squeeze_first_dim:
        shap_values = shap_values[0]

    return shap_values

def get_op_idx_by_format(idx_arr):

    combo_arr = np.empty((idx_arr.size + idx_arr.size,), dtype=idx_arr.dtype)
    combo_arr[0::2] = idx_arr
    combo_arr[1::2] = idx_arr

    return combo_arr


def make_custom_montage(montage_path):

    ''' make custom montage from EGI 128 template and drop electrodes
        125-130 for consistency with preprocessing pipeline'''

    montage = mne.channels.read_custom_montage(montage_path) # Or use your existing montage

    # reset channel names
    montage.ch_names = [str(ch) for ch in list(range(0,130))]

    # Define the channels to remove
    channels_to_remove = [str(ch) for ch in list(range(125,130))]

    # Create a new list of dig_points *and* corresponding ch_names
    # This ensures that both lists are always in sync
    filtered_eeg_dig_points = []
    filtered_eeg_ch_names = []
    fiducial_dig_points = []

    for i, dp in enumerate(montage.dig):
        # Assuming 'ident' within dp.items().mapping holds the channel name/identifier
        ch_name_from_dig = str(dp.items().mapping['ident'])
        
        if dp['kind'] == FIFF.FIFFV_POINT_EEG: # Check if it's an EEG point
            if ch_name_from_dig not in channels_to_remove:
                filtered_eeg_dig_points.append(dp)
                filtered_eeg_ch_names.append(ch_name_from_dig)
                
        elif dp['kind'] in (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_RPA):
            # Keep fiducials if needed, but don't include their names in ch_names for DigMontage
            fiducial_dig_points.append(dp)
        # You might want to handle other point types (e.g., FIFFV_POINT_HPI, FIFFV_POINT_EXTRA) if present

    # Combine the filtered EEG points and the fiducial points for the new DigMontage
    # The 'dig' argument can contain both, but 'ch_names' should only be for EEG
    all_filtered_dig_points = fiducial_dig_points + filtered_eeg_dig_points

    # Create a new DigMontage from the filtered data
    filtered_montage = mne.channels.DigMontage(dig=all_filtered_dig_points, ch_names=filtered_eeg_ch_names)

    return filtered_montage


def make_epochs(dataset, info, cat, event_dict, n_trials, idx_filt):
    """
    Create an MNE EpochsArray from preprocessed dataset trials.
    
    Takes a preprocessed dataset and converts trials of a specified category
    into an MNE EpochsArray object for further analysis.
    
    Parameters
    ----------
    dataset : dict
        Preprocessed dataset containing:
        - 'X_3d': 3D array of EEG data with shape (channels, timepoints, trials)
        - Category data (e.g., 'labels_correct') with trial labels
    info: dict
        MNE Info Object
    cat : array-like
        Category extracted from dataset (e.g., dataset['labels_correct']).
        Special handling for 'labels_correct' vs other categories.
    event_dict : dict
        Mapping of event names to event codes for MNE compatibility.
    n_trials : int
        Total number of trials in the dataset.
    idx_filt : array-like
        Boolean or integer array for filtering specific trials.
        Not applied when cat='labels_correct'.
    
    Returns
    -------
    mne.EpochsArray
        MNE EpochsArray object with:
        - Baseline correction applied (-150ms to 0ms)
        - Time window starting at -150ms
        - Events and event_id properly formatted
    
    Notes
    -----
    - Assumes 1-second intervals between trial onsets
    - Applies baseline correction from -150ms to stimulus onset (0ms)
    - Transposes data from (channels, timepoints, trials) to 
      (trials, channels, timepoints) for MNE compatibility
    """

    events = np.column_stack(
        (
            np.arange(0, int(info['sfreq'])*n_trials,int(info['sfreq'])),
            np.zeros(n_trials, dtype=int),
            cat,
        )
    ) 
    
    epochs = mne.EpochsArray(data=np.transpose(dataset['X_3d'][:,:,idx_filt], (2, 0, 1)),
                                        info=info, tmin=-0.150, baseline=(-0.150, 0),
                                        events=events, event_id=event_dict)

    return epochs


def run_LDA(epoch, categories, cv=5):

    cat_pairs = list(itertools.combinations(categories, 2))
    results = {}

    for pair in cat_pairs:

        results[pair] = dict(coef_list=[], score_list=[])

        # Parameters
        cross_val = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # To collect results
        scores = []
        coefs = []

        X = epoch[[pair[0],pair[1]]].get_data(copy=False)  # EEG signals: n_epochs, n_meg_channels, n_times
        y = epoch[[pair[0],pair[1]]].events[:, 2]  # target: auditory left vs visual left
        
#         scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4)

        # Mean scores across cross-validation splits
#         score = np.mean(scores, axis=0)

        for train_idx, test_idx in cross_val.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=0.99),
#                 LinearModel(LogisticRegression(solver="lbfgs")
                LinearModel(LinearDiscriminantAnalysis(solver="lsqr"))
            )

            time_decod = SlidingEstimator(clf, n_jobs=4, scoring="accuracy", verbose=True)
            time_decod.fit(X_train, y_train)

            coefs.append(get_coef(time_decod, "patterns_", inverse_transform=True))
            scores.append(time_decod.score(X_test, y_test))

        mean_coef = np.array(coefs).mean(axis=0)
        mean_scores = np.array(scores).mean(axis=0)

        results[pair]['coef_list'] = mean_coef
        results[pair]['score_list'] = mean_scores

        return results
    

def run_LDA_gen(train_epoch, test_epoch, categories):

    cat_pairs = list(itertools.combinations(categories, 2))
    results = {}

    for pair in cat_pairs:

        results[pair] = dict(coef_list=[], score_list=[])

        X_train = train_epoch[[pair[0],pair[1]]].get_data(copy=False)  # EEG signals: n_epochs, n_meg_channels, n_times
        y_train = train_epoch[[pair[0],pair[1]]].events[:, 2]  # target: auditory left vs visual left
        
        X_test = test_epoch[[pair[0],pair[1]]].get_data(copy=False)  # EEG signals: n_epochs, n_meg_channels, n_times
        y_test = test_epoch[[pair[0],pair[1]]].events[:, 2]  # target: auditory left vs visual left

        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.99),
#                 LinearModel(LogisticRegression(solver="lbfgs"))
            LinearModel(LinearDiscriminantAnalysis(solver="lsqr"))
        )

        time_decod = SlidingEstimator(clf, n_jobs=4, scoring="accuracy", verbose=True)
        time_decod.fit(X_train, y_train)

        results[pair]['coef_list'] = get_coef(time_decod, "patterns_", inverse_transform=True)
        results[pair]['score_list'] = time_decod.score(X_test, y_test)

        return results
    
    

