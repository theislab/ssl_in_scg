#!/bin/bash

#SBATCH -J VAE_ind_gene_masking_gridsearch
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


sbatch 'VAE_50p_v1.sh' &
sbatch 'VAE_50p_v2.sh' &
sbatch 'VAE_50p_v3.sh' &
sbatch 'VAE_50p_v4.sh' &
# sbatch 'VAE_50p_v5.sh' &
# sbatch 'VAE_50p_v6.sh' &
# sbatch 'VAE_50p_v7.sh' &
# sbatch 'VAE_50p_v8.sh' &
wait


