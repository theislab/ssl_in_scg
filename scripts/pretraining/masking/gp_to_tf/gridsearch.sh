#!/bin/bash

#SBATCH -J gp_to_tf_masking_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'MLP_v1.sh' &
sbatch 'MLP_v2.sh' &
sbatch 'MLP_v3.sh' &
sbatch 'MLP_v4.sh' &
sbatch 'MLP_v5.sh' &
sbatch 'MLP_v6.sh' &
sbatch 'MLP_v7.sh' &
sbatch 'MLP_v8.sh' &
sbatch 'MLP_v9.sh' &
sbatch 'MLP_v10.sh' &
sbatch 'MLP_v11.sh' &
sbatch 'MLP_v12.sh' &
wait


