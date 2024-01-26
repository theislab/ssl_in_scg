#!/bin/bash

#SBATCH -J gp_hvg_mask_ae_10p
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --mask_rate 0.1 --masking_strategy 'gene_program' --model 'MLP' --dropout 0.1 --weight_decay 0.01 --lr 0.001 --decoder True --gp_file 'C8' --hvg True --missing_tolerance 10