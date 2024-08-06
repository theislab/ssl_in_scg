#!/bin/bash

#SBATCH -J VAE_HLCA_Recon_No_SSL_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


sbatch 'VAE_v1.sh' &
sbatch 'VAE_v2.sh' &
sbatch 'VAE_v3.sh' &
sbatch 'VAE_v4.sh' &
wait


