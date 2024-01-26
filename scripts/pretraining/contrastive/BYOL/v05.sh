#!/bin/bash

#SBATCH -J contr_v5
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

python -u train.py --augment_intensity=0.01 --augment_type='Gaussian' --model='MLP' --lr=0.00005 --contrastive_method='BYOL' --version='v5' --weight_decay=0.0 --dropout=0.0 --batch_size=8192 --model_path=/lustre/groups/ml01/workspace/$USER/
