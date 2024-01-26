#!/bin/bash

#SBATCH -J Pretrain_GP_to_TF_run1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --masking_strategy='gp_to_tf' --model='MLP' --dropout=0.05 --weight_decay=0.0 --lr=0.001 --gp_file='C3' --missing_tolerance=90 --decoder --version='run1' --batch_size=16384
