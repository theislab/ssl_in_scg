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


sbatch 'v01.sh' &
sbatch 'v03.sh' &
sbatch 'v07.sh' &
sbatch 'v08.sh' &
sbatch 'v10.sh' &
wait


