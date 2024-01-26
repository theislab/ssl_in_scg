#!/bin/bash

#SBATCH -J rec_MLP_gridsearch
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


sbatch 'HVG_MLP_v1.sh' &
sbatch 'HVG_MLP_v2.sh' &
sbatch 'HVG_MLP_v3.sh' &
sbatch 'HVG_MLP_v4.sh' &
wait


