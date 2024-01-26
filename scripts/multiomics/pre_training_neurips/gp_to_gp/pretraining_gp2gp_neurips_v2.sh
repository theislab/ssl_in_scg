#!/bin/bash

#SBATCH -J gp2gp_v2
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train.py --masking_strategy 'gp_to_gp' --model 'MAE' --version 'GP_to_GP_Pretrain' --mode 'pre_training' --dropout 0.0 --weight_decay 0.00001 --learning_rate 0.005
