#!/bin/bash

#SBATCH -J run_all_ind_masked_HVG_MLP
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=10GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'MLP_10p.sh' &
sbatch 'MLP_20p.sh' &
sbatch 'MLP_30p.sh' &
sbatch 'MLP_40p.sh' &
sbatch 'MLP_50p.sh' &
sbatch 'MLP_60p.sh' &
sbatch 'MLP_70p.sh' &
sbatch 'MLP_80p.sh' &
sbatch 'MLP_90p.sh' &
wait


