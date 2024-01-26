#!/bin/bash

#SBATCH -J MLP_HVG_Analysis_Pretrain
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


sbatch 'HVG_MLP_50p_1000.sh' "$1" &
sbatch 'HVG_MLP_50p_3000.sh' "$1" &
sbatch 'HVG_MLP_50p_4000.sh' "$1" &
sbatch 'HVG_MLP_50p_5000.sh' "$1" &
sbatch 'HVG_MLP_50p_6000.sh' "$1" &
sbatch 'HVG_MLP_50p_7000.sh' "$1" &
sbatch 'HVG_MLP_50p_8000.sh' "$1" &
sbatch 'HVG_MLP_50p_9000.sh' "$1" &
sbatch 'HVG_MLP_50p_10000.sh' "$1" &
wait


