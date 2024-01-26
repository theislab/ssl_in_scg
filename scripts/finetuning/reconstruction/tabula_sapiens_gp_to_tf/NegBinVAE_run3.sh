#!/bin/bash

#SBATCH -J TabSab_NegBinVAE_Recon_GP_to_TF_run2
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/

python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=8192 --version='new_run3' --model 'NegBinVAE' --dropout=0.05 --stochastic --supervised_subset="Tabula_Sapiens" --pretrained_dir="/lustre/groups/ml01/workspace/$USER/trained_models/pretext_models/masking/CN_MLP_gp_to_tf/default/version_2/checkpoints/best_checkpoint_val.ckpt"