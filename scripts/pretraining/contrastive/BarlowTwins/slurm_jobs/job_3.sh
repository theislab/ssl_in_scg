#!/bin/bash

#SBATCH -J bt_contr_3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=80GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

python -u train.py --negbin_intensity 0.7 --dropout_intensity 0.4 --p 0.2 --contrastive_method "bt"
