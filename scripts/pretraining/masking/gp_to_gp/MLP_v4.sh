#!/bin/bash

#SBATCH -J only_gp_v4
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 12:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --masking_strategy='single_gene_program' --model='MLP' --dropout=0.0 --weight_decay=0.00001 --lr=0.005 --decoder --gp_file='C8' --version='v4a'
