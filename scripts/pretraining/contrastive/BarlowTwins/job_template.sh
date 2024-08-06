#!/bin/bash

#SBATCH -J bt_contr_{index}
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

python -u train.py --negbin_intensity {negbin_intensity} --dropout_intensity {dropout_intensity} --p {p} --contrastive_method "bt"
