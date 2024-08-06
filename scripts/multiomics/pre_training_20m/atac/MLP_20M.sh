#!/bin/bash

#SBATCH -J ATAC_Pretrain_20M_Ind_Mask
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=28
#SBATCH -t 2-00:00:00
#SBATCH --mem=240GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi



cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train_multi.py --masking_strategy 'random' --dataset "20M" --model 'MAE' --mode 'pre_training' --dropout 0.1 --weight_decay 0.01 --learning_rate 0.001 --batch_size 8192 --version "tfidf"
