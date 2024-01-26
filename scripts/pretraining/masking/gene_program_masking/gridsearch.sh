#!/bin/bash

#SBATCH -J gp_mask_MLP_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'MLP_01p_v1.sh' &
sbatch 'MLP_01p_v2.sh' &
sbatch 'MLP_01p_v3.sh' &
sbatch 'MLP_01p_v4.sh' &
sbatch 'MLP_01p_v5.sh' &
sbatch 'MLP_01p_v6.sh' &
sbatch 'MLP_01p_v7.sh' &
sbatch 'MLP_01p_v8.sh' &
sbatch 'MLP_01p_v9.sh' &
sbatch 'MLP_01p_v10.sh' &
sbatch 'MLP_01p_v11.sh' &
sbatch 'MLP_01p_v12.sh' &
wait


