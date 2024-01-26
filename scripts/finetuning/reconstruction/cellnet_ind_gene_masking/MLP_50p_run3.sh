#!/bin/bash

#SBATCH -J Rec_MLP_Ind_Masking_run3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/

python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=8192 --version='run3' --dropout=0.05 --stochastic --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/masking/CN_MLP_50p/default/version_1/checkpoints/best_checkpoint_val.ckpt"