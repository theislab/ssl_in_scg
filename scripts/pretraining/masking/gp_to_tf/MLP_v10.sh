#!/bin/bash

#SBATCH -J gp_to_tf_mask_ae_01p_v10
#SBATCH -p gpu_p
#SBATCH --qos gpu
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

python -u train.py --masking_strategy 'gp_to_tf' --model 'MLP' --version 'v10' --dropout 0.1 --weight_decay 0.1 --lr 0.0001 --decoder False --gp_file 'C3' --hvg True --missing_tolerance 10