#!/bin/bash

#SBATCH -J HLCA_Recon_No_SSL_VAE_v3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 08:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/

python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=8192 --version='v3' --model 'VAE' --vae_type 'simple_vae' --dropout=0.1 --stochastic --supervised_subset="HLCA"