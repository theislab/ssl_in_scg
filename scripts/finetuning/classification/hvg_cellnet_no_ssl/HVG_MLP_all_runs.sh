#!/bin/bash

#SBATCH -J MLP_gridsearch
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


sbatch 'HVG_MLP_run0.sh' &
sbatch 'HVG_MLP_run1.sh' &
sbatch 'HVG_MLP_run2.sh' &
sbatch 'HVG_MLP_run3.sh' &
sbatch 'HVG_MLP_run4.sh' &
wait


