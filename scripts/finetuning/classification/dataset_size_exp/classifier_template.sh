#!/bin/bash

#SBATCH -J Clf_{supervised_subset}_{pretrained_model_name}
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/classifier

python -u train.py --version="{version}" --stochastic --supervised_subset="{supervised_subset}" --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/{pretrained_dir}"
