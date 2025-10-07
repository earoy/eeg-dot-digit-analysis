import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import path as op

def load_data(filename, data_dir, n_blocks=4):

    data_dict = sp.io.loadmat(op.join(data_dir, filename))

    if data_dir[-1]=='1':
        eeg_data = data_dict['classData'][0][0][0][0][0][0]
    else:
        eeg_data = data_dict['classData'][0][0][0]

    n_samples = eeg_data.shape[1]
    n_epochs = eeg_data.shape[2]

    # 
    block_mask = np.zeros(n_epochs, dtype=bool)
    block_mask[:200*n_blocks] = True
    correct_block_mask = np.zeros(int(n_epochs/5), dtype=bool)
    correct_block_mask[:40*n_blocks] = True

    eeg_data = eeg_data[:,:,block_mask]

    # 1: dots 2: digits, 3: math symbol
    good_epochs = data_dict['classData'][0][0][1]
    good_epochs = np.array([int(i[0]) for i in good_epochs])
    good_epochs = good_epochs[block_mask]

    # 1: dots 2: digits, 3: math symbol
    labels3 = data_dict['classData'][0][0][2]
    labels3 = np.array([int(i[0]) for i in labels3])
    labels3 = labels3[block_mask]

    # 1 for dot trial, 2 for digit trial
    trial_format = labels3[np.arange(0, len(labels3), 5)]

    # values 1-7 match numerosity 100: math symbol
    labels_numerosity = data_dict['classData'][0][0][3]
    labels_numerosity = np.array([int(i[0]) for i in labels_numerosity])
    labels_numerosity = labels_numerosity[block_mask]

    # generate labels for numerosity+format combo
    labels_num_form = labels_numerosity.astype('str') + labels3.astype('str')
    labels_num_form = labels_num_form.astype(int)

    # 1: subitizing 2: counting, 3: math symbol
    labels_range = data_dict['classData'][0][0][4]
    labels_range = np.array([int(i[0]) for i in labels_range])
    labels_range = labels_range[block_mask]

    # 0: incorrect 1: correct
    labels_correct = data_dict['classData'][0][0][5]
    labels_correct = np.array([int(i[0]) for i in labels_correct])
    labels_correct = labels_correct[correct_block_mask]

    # Compute the indices of the first and third elements in each chunk (each operand)
    n = 5  # Chunk size
    indices_first = np.arange(0, len(labels_numerosity), 5)
    indices_third = np.arange(2, len(labels_numerosity), 5)
    indices_fifth = np.arange(4, len(labels_numerosity), 5)

    # Combine indices
    operand_idx = np.sort(np.append(indices_first,indices_third))

    # operands and figure out the sums by adding every 2
    labels_operands = labels_numerosity[operand_idx]
    labels_sum = np.array([int(sum(labels_operands[i:i+2])) for i in range(0, len(labels_operands) - len(labels_operands)%2, 2)])

    # Get index for equals sign trials
    eq_idx = np.arange(3, len(labels_numerosity), n) 

    # Get index for solution trials
    solution_idx = np.arange(4, len(labels_numerosity), n) 

    dims = eeg_data.shape
    X = eeg_data.reshape(dims[0] * dims[1], dims[2]).T

    # extract EEG data related to operands and combine
    eeg_data_numbers = eeg_data[:,:,operand_idx]
    eeg_operands = np.hstack([eeg_data_numbers[:, :, ::2], eeg_data_numbers[:, :, 1::2]])
    dims = eeg_data.shape
    X_operands = eeg_operands.reshape(dims[0] * (2*dims[1]), int(dims[2]/5)).T

    return {'X': X,
            'X_3d': eeg_data,
            'X_multi': X_operands,
            'n_samples': n_samples,
            'good_epochs': good_epochs,
            'labels3': labels3,
            'labels_numerosity':labels_numerosity,
            "labels_num_form": labels_num_form,
            "labels_range":labels_range,
            "labels_sum": labels_sum,
            "labels_correct": labels_correct,
            "eq_idx": eq_idx,
            "op_1_idx": indices_first,
            "op_2_idx": indices_third,
            "solution_idx": solution_idx,
            "trial_format": trial_format
            }

def plot_AM(data, metric, vmin=0.5, vmax=0.8):
    """
    Helper function to plot accuracy matrices after fitting models.

    Works for keys like:
      '1_2'
      '11_dot_12_digit'
      '12_digit_42_digit'
    """
    # --- Step 1: Extract all unique labels ---
    all_labels = sorted({part for key in data.keys() for part in key.split('_')})
    
    # Map label -> matrix index
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    size = len(all_labels)

    # --- Step 2: Initialize matrix ---
    matrix = np.full((size, size), np.nan)

    # --- Step 3: Fill the matrix ---
    for key, metrics in data.items():
        parts = key.split('_')
        if len(parts) != 2:
            # If your key has more than two underscores, we assume it's like "label1_label2"
            # But each label might have underscores itself, so split from the right.
            # Actually, safest is to split exactly once at the middle occurrence.
            # But your provided keys always seem to have exactly two big parts joined by "_".
            pass
        val1, val2 = parts[0], parts[1]
        idx1, idx2 = label_to_idx[val1], label_to_idx[val2]
        
        val = metrics[metric]
        matrix[idx1, idx2] = val
        matrix[idx2, idx1] = val  # symmetric

    # --- Step 4: Make DataFrame ---
    df = pd.DataFrame(matrix, index=all_labels, columns=all_labels)

    # --- Step 5: Plot ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=vmin, vmax=vmax, linewidths=0.5)
    plt.title(f"Pairwise Accuracy Heatmap ({metric})")
    plt.show()

    return df