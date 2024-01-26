#!/bin/bash

#SBATCH -J Pert_Sciplex_No_SSL_v10
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 4:00:00
#SBATCH --mem=160GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/perturbations/

python -u train.py --batch_size 8192 --version 'v10' --lr 1e-4 --weight_decay 1e-5 --dropout 1e-3