#!/bin/bash

#SBATCH -J Pretrain_Data_0.01_seed_3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --mask_rate 0.5 --model 'MLP' --dropout 0.1 --weight_decay 0.01 --lr 0.001 --decoder --version 'data_0.01_seed_3' --donor_list /lustre/groups/ml01/workspace/till.richter/scTab/donor_id_subset_0.01_seed_3.npy
