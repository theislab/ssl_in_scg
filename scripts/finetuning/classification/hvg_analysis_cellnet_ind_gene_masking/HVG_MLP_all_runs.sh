#!/bin/bash

#SBATCH -J MLP_HVG_Analysis
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


sbatch 'HVG_MLP_1000.sh' "$1" &
sbatch 'HVG_MLP_2000.sh' "$1" &
sbatch 'HVG_MLP_3000.sh' "$1" &
sbatch 'HVG_MLP_4000.sh' "$1" &
sbatch 'HVG_MLP_5000.sh' "$1" &
sbatch 'HVG_MLP_6000.sh' "$1" &
sbatch 'HVG_MLP_7000.sh' "$1" &
sbatch 'HVG_MLP_8000.sh' "$1" &
sbatch 'HVG_MLP_9000.sh' "$1" &
sbatch 'HVG_MLP_10000.sh' "$1" &
wait


