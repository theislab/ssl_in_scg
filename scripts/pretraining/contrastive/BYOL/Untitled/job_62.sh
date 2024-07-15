#!/bin/bash

#SBATCH -J contr_62
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

python -u train.py --p 0.5 --negbin_intensity 0.2 --dropout_intensity 0.05 --lr 0.01 --weight_decay 1e-06 --max_epochs 12
