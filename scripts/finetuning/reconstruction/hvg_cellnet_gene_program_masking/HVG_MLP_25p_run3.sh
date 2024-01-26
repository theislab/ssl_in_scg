#!/bin/bash

#SBATCH -J Rec_HVG_MLP_Gene_Program_Masking_25p_run3
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
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/
python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=16384 --version='run3' --hvg --dropout=0.05 --stochastic --pretrained_dir="/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/masking/CN_HVG_2000_MLP_gene_program_C8_25p/default/version_2/checkpoints/best_checkpoint_val.ckpt"