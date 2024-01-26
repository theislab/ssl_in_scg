#!/bin/bash

#SBATCH -J Multi_Pretrain_NeurIPS_GP
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=10
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate celldreamer




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

python train.py --masking_strategy 'gene_program' --model 'MAE' --mode 'pre_training' --dropout  0.1652797684055134 --weight_decay 0.0011654517196319286 --learning_rate 0.00039656945603882226 --batch_size 256

