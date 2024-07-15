#!/bin/bash

#SBATCH -J Supervised_Finetune_NeurIPS_run4
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH --mem=80GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate celldreamer



cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics


python train_multi.py --mode 'fine_tuning' --version 'new2_big_tfidf_run4' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05


