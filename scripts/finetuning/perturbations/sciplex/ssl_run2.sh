#!/bin/bash

#SBATCH -J Pert_Sciplex_SSL_run2
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=160GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/perturbations/

python -u train.py --batch_size 8192 --version 'easy_clf_ssl_run2' --lr 1e-4 --weight_decay 1e-5 --dropout 1e-2 --pretrained_dir '/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/masking/CN_Pert_MLP_50psciplex2020/default/version_0/checkpoints/best_checkpoint_val.ckpt'