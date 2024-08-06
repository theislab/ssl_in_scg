#!/bin/bash

#SBATCH -J bt_contr_9
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH --mem=80GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

python -u train.py --negbin_intensity 0.2 --dropout_intensity 0.2 --lr 0.01 --max_epochs 12 --contrastive_method "bt"
