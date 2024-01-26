#!/bin/bash

#SBATCH -J VAE_50p_v5
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 8:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --mask_rate=0.5 --model 'VAE' --vae_type 'scvi_vae' --version 'v5' --dropout 0.1 --weight_decay 0.1 --lr 0.001