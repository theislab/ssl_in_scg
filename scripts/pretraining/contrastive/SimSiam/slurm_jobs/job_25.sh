#!/bin/bash

#SBATCH -J SimSiam_25
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

python -u train.py --p 0.7 --negbin_intensity 0.4 --dropout_intensity 0.2 --max_epochs 12 --contrastive_method "SimSiam"
