#!/bin/bash

#SBATCH -J Supervised_Finetune_NeurIPS
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7



cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics

# python train.py --mode 'fine_tuning' --version 'run0' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05
python train.py --mode 'fine_tuning' --version 'run1' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05
python train.py --mode 'fine_tuning' --version 'run2' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05
python train.py --mode 'fine_tuning' --version 'run3' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05
python train.py --mode 'fine_tuning' --version 'run4' --dropout 0.15740626023302481 --weight_decay 0.0011635827043670439 --learning_rate 8.156092178310426e-05


