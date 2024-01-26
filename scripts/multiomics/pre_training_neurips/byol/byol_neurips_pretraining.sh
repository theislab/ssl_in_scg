#!/bin/bash

#SBATCH -J Multi_Pretrain_NeurIPS_BYOL
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
python train.py --version 'bt_pretrain_neurips' --model 'BYOL' --mode 'pre_training' --batch_size 4096 --dropout 0.0 --weight_decay 1e-6


