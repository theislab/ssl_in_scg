#!/bin/bash

#SBATCH -J FineTuning_Multiomics_20M_random_MAE
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
python train.py --mode 'fine_tuning' --version 'run9' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_random_MAE_Ind_Mask_Pretrain_20M_v1_Test2a/default/version_0/checkpoints/best_checkpoint_val.ckpt
