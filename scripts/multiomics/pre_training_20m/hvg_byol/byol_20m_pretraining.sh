#!/bin/bash

#SBATCH -J Multi_Pretrain_20M_BYOL
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=20
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile_prak7"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
cd /home/icb/till.richter/self_supervision/self_supervision/trainer/multiomics
python train.py --version 'new_bt_pretrain_20m' --dataset "20M" --model 'BYOL' --mode 'pre_training' --dropout 0.0 --weight_decay 0.0  --learning_rate=0.00005 --batch_size 4096


