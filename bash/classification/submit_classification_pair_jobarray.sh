#!/bin/bash
#
#SBATCH --job-name=MatClassRSA
#SBATCH --partition=normal,gse
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-3

ml load matlab

#CONFIG_FILES='config_S06.txt'
#CONFIG_FILES='config_pairwise_all.txt'
CONFIG_FILES='config_answer.txt'
export function_call=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${CONFIG_FILES} )

echo $function_call

matlab -nodisplay < \
/home/users/ethanroy/eeg_groupitizing/code/matlab/runRSA_job.m

