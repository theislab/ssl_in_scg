#!/bin/bash

#SBATCH -J MLP_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'MLP_v5.sh' &
sbatch 'MLP_v6.sh' &
sbatch 'MLP_v7.sh' &
sbatch 'MLP_v8.sh' &
wait


