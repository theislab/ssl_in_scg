#!/bin/bash

#SBATCH -J ATAC_Pretrain_NeurIPS_Ind_Mask
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi



cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train_multi.py --masking_strategy 'random' --dataset "NeurIPS" --model 'MAE' --mode 'pre_training' --dropout 0.1 --weight_decay 0.01 --learning_rate 0.001 --batch_size 8192 --version "tfidf"
