#!/bin/bash

#SBATCH -J Multi_Pretrain_20M_GP_v1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=20
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile_prak7"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
python train.py --masking_strategy 'gene_program' --dataset "20M" --version 'GP_Pretrain_20M_v1' --model 'MAE' --mode 'pre_training' --dropout 0.1 --batch_size 8192 --weight_decay 0.01 --learning_rate 0.001
