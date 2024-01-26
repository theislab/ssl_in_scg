#!/bin/bash

#SBATCH -J Pert_Sciplex_No_SSL_run0
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 05:00:00
#SBATCH --mem=80GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/perturbations/
python -u scgen_train.py --batch_size 16384 --version 'run2' --lr 1e-5 --weight_decay 1e-5 --dropout 1e-2 --model_type 'Unsupervised' --pretrained_dir '/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/masking/CN_Pert_MLP_50pFullSplit_sciplex2020/default/version_0/checkpoints/best_checkpoint_val.ckpt'
