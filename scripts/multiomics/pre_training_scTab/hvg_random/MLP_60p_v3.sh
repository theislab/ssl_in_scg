#!/bin/bash

#SBATCH -J Multi_Pretrain_20M_Ind_Mask_v3
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=28
#SBATCH -t 2-00:00:00
#SBATCH --mem=240GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile_prak7"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train.py --masking_strategy 'random' --dataset "20M" --version 'Ind_Mask_Pretrain_20M_v1_Test3a' --model 'MAE' --mode 'pre_training' --dropout 0.0 --weight_decay 0.0 --learning_rate 0.0001 --batch_size 8192
