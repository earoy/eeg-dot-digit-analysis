#!/bin/bash
#
#SBATCH --job-name=ShortEpochCleaning
#SBATCH --partition=normal,gse
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-2

ml load matlab/R2023b 

CONFIG_FILES='configs_208.txt'
export config_path=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${CONFIG_FILES} )

echo $config_path

matlab -nodisplay < \
~/eeg_groupitizing/code/matlab/preproc/runCleaning.m

# use this to update the config file: ls configs/ > configs.txt
# use this to get array length: ls configs | wc -l
