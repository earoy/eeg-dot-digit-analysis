#!/bin/bash
#
#SBATCH --job-name=EpochClassification
#SBATCH --partition=normal,gse
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-17

SUBJECTS_FILE='/home/users/ethanroy/eeg_groupitizing/code/bash/preproc/eni_list.txt'

subject=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${SUBJECTS_FILE} )
echo ${PWD}

in_dir=/scratch/users/ethanroy/eeg_groupitizing_data/classification_input/
out_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/classification_results/
montage=/home/users/ethanroy/eeg_groupitizing/code/python/classification/utils/2_9AverageNet128_v1.sfp
echo $subject
echo $in_dir
echo $out_dir


conda run -v -n mne python3 -u ~/eeg_groupitizing/code/python/classification/epoch_classification.py \
$subject $in_dir $out_dir 

conda run -v -n mne python3 -u ~/eeg_groupitizing/code/python/classification/mne_classification.py \
$subject $montage $out_dir 

#in_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/classification_results/
#out_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/perm_results/
#conda run -v -n mne python3 -u ~/eeg_groupitizing/code/python/classification/perm_correct_classification.py \
#$subject $in_dir $out_dir 
