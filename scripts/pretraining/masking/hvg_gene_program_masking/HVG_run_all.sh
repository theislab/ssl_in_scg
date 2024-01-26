#!/bin/bash

#SBATCH -J run_all_gp_masks
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --mem=10GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'HVG_MLP_01p.sh' &
sbatch 'HVG_MLP_02p.sh' &
sbatch 'HVG_MLP_03p.sh' &
sbatch 'HVG_MLP_04p.sh' &
sbatch 'HVG_MLP_05p.sh' &
sbatch 'HVG_MLP_10p.sh' &
sbatch 'HVG_MLP_15p.sh' &
sbatch 'HVG_MLP_20p.sh' &
sbatch 'HVG_MLP_25p.sh' &
sbatch 'HVG_MLP_50p.sh' &
wait


