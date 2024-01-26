#!/bin/bash

#SBATCH -J contr_v3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 12:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

python -u train.py --augment_intensity=0.001 --learning_rate_weights=0.02 --weight_decay=1e-6 --contrastive_method='bt' --version='v3' --batch_size=4096 --hvg --model_path=/lustre/groups/ml01/workspace/$USER/
