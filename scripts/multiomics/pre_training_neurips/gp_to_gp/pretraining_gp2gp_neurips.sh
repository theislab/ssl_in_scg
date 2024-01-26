#!/bin/bash

#SBATCH -J Multi_Pretrain_NeurIPS_GP_GP
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=10
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train.py --masking_strategy 'gp_to_gp' --model 'MAE' --version 'GP_to_GP_Pretrain' --mode 'pre_training' --dropout 0.10197817839106406 --weight_decay 0.008720560533726803 --learning_rate 1.8820976052295424e-05 --batch_size 16384
