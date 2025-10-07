#!/bin/bash
#
#SBATCH --job-name=perm_test
#SBATCH --partition=normal,gse
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-17


SUBJECTS_FILE='/home/users/ethanroy/eeg_groupitizing/code/bash/preproc/eni_list.txt'

subject=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${SUBJECTS_FILE} )
echo ${PWD}

in_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/classification_results/
out_dir=/scratch/users/ethanroy/eeg_groupitizing_data/results/perm_results/
echo $subject
echo $in_dir
echo $out_dir

conda run -v -n mne python3 -u ~/eeg_groupitizing/code/python/classification/run_classification_perm.py \
$subject $in_dir $out_dir 

