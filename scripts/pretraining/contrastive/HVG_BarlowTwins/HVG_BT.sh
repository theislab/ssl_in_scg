#!/bin/bash

#SBATCH -J byol_best_setting
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
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

#python -u train.py --augment_intensity=0.001 --augment_type='Gaussian' --model='MLP' --lr=0.0001 --contrastive_method='BYOL' --weight_decay=0.0 --dropout=0.0 --hvg True --batch_size=8192 --hvg
python -u train.py --augment_intensity=0.001 --learning_rate_weights=0.2 --weight_decay=1e-6 --contrastive_method='bt' --batch_size=8192 --hvg --model_path=/lustre/groups/ml01/workspace/$USER/
