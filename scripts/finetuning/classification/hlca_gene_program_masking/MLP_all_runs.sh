#!/bin/bash

#SBATCH -J MLP_Clf_GP_Masking
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=10GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'MLP_50p_run0.sh' &
sbatch 'MLP_50p_run1.sh' &
sbatch 'MLP_50p_run2.sh' &
sbatch 'MLP_50p_run3.sh' &
sbatch 'MLP_50p_run4.sh' &
wait


