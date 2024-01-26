#!/bin/bash

#SBATCH -J hvg_only_gp_mask_best
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=240GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --masking_strategy 'single_gene_program' --model 'MLP' --dropout 0.05 --weight_decay 0.0 --lr 0.001 --gp_file 'C8' --hvg --missing_tolerance 33 --decoder --version 'run0' --batch_size=32768
