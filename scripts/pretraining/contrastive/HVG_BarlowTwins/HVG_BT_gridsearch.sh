#!/bin/bash

#SBATCH -J contr_MLP_gridsearch
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


sbatch 'HVG_v01.sh' &
sbatch 'HVG_v03.sh' &
sbatch 'HVG_v07.sh' &
sbatch 'HVG_v08.sh' &
sbatch 'HVG_v10.sh' &
wait


