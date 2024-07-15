#!/bin/bash

#SBATCH -J HLCA_Recon_No_SSL_run3
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=240GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile"
fi





cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/

python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=16384 --model 'NegBinVAE' --version='new_run3' --dropout=0.05 --stochastic --supervised_subset="HLCA"