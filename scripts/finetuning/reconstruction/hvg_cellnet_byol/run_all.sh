#!/bin/bash

#SBATCH -J MLP_Rec_BT_all_runs
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch MLP_run0.sh &
sbatch MLP_run1.sh &
sbatch MLP_run2.sh &
sbatch MLP_run3.sh &
sbatch MLP_run4.sh &
wait