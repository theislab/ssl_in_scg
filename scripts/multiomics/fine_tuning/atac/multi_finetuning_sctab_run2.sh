#!/bin/bash

#SBATCH -J Supervised_Finetune_scTab_run2
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
conda activate celldreamer



cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics


python train_multi.py --mode 'fine_tuning' --version 'new2_big_tfidf_run2' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05 --pretrained_dir "/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/atac_multiomics_20M_random_MAE_tfidf/default/version_0/checkpoints/best_checkpoint_val.ckpt"

