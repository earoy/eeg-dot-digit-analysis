#!/bin/bash
#
#SBATCH --job-name=EpochClassification
#SBATCH --partition=normal,gse
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-3

SUBJECTS_FILE='subject_epoch_class_test.txt'

subject=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${SUBJECTS_FILE} )
echo ${PWD}

in_dir=/scratch/users/ethanroy/eeg_groupitizing_data/classification_input/
out_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/classification_results/
echo $subject
echo $in_dir
echo $out_dir

conda run -v -n mne python3 -u ~/eeg_groupitizing/code/python/classification/mne_analysis.py $subject $in_dir $out_dir 
