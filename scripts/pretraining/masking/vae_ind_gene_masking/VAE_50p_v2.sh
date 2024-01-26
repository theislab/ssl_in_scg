#!/bin/bash

#SBATCH -J VAE_50p_best
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

python -u train.py --mask_rate=0.5 --model 'VAE' --vae_type 'simple_vae' --version '' --dropout 0.1 --weight_decay 0.1 --lr 0.0001 --decoder --batch_size 16384