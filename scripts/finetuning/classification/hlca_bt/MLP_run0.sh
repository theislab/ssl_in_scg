#!/bin/bash

#SBATCH -J clf_hlca_bt_run0
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/classifier

CUBLAS_WORKSPACE_CONFIG=:16:8 python -u train.py --version="run0" --lr=9e-4 --dropout=0.1 --weight_decay=0.05 --stochastic --supervised_subset="HLCA" --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/contrastive/MLP_bt_Gaussian_0_01/best_checkpoint_val.ckpt"  --model_path=/lustre/groups/ml01/workspace/$USER/
