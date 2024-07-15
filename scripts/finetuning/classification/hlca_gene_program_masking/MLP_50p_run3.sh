#!/bin/bash

#SBATCH -J MLP_Clf_GP_Masking_run3
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

CUBLAS_WORKSPACE_CONFIG=:16:8 python -u train.py --version="run3" --lr=9e-4 --dropout=0.0 --weight_decay=0.05 --supervised_subset="HLCA" --stochastic --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/masking/CN_MLP_gene_program_C8_50p/default/version_1/checkpoints/best_checkpoint_val.ckpt"
