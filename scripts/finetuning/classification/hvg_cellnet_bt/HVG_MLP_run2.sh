#!/bin/bash

#SBATCH -J MLP_HVG_Clf_BT_run2
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
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

CUBLAS_WORKSPACE_CONFIG=:16:8 python -u cellnet_mlp.py --version="run2" --lr=9e-4 --dropout=0.0 --hvg --weight_decay=0.05 --stochastic --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/contrastive/HVG_2000_MLP_bt_Gaussian_0_001/best_checkpoint_val.ckpt"
