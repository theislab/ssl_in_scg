#!/bin/bash

#SBATCH -J hvg_cn_mlp_byol_run1
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/classifier

CUBLAS_WORKSPACE_CONFIG=:16:8 python -u train.py --version="run1" --lr=9e-4 --dropout=0.0 --weight_decay=0.05 --hvg=True --stochastic=True --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/contrastive/CN_HVG_MLP_BYOL_Gaussian_0.001_/default/version_0/checkpoints/best_checkpoint_val.ckpt"
