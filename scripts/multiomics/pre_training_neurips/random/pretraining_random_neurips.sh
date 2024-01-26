#!/bin/bash

#SBATCH -J Multi_Pretrain_NeurIPS_Random
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

python train.py --masking_strategy 'random' --version 'Ind_Mask_Pretrain' --model 'MAE' --mode 'pre_training' --dropout 0.11468766849892434 --weight_decay 0.0010896018112943387 --learning_rate 0.0003312628014499169 --batch_size 16384