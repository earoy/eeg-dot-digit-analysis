#!/bin/bash
#SBATCH --job-name=transfer_data
#SBATCH --time=48:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --mail-user=sagon151@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=normal,gse

ml load matlab/R2023b 

matlab -nodisplay < \
/home/users/sagon151/eeg-dot-digit-analysis/matlab/preproc/setup_classifcation_data_sherlock.m
