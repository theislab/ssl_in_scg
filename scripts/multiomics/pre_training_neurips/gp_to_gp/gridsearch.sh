#!/bin/bash

#SBATCH -J only_gp_masking_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7
sbatch 'pretraining_gp2gp_neurips_v1.sh' &
sbatch 'pretraining_gp2gp_neurips_v2.sh' &
sbatch 'pretraining_gp2gp_neurips_v3.sh' &
wait


