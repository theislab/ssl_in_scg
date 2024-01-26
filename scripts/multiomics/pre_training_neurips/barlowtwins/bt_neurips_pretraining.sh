#!/bin/bash

#SBATCH -J gp_to_tf_masking_gridsearch
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
<<<<<<< Updated upstream:scripts/multiomics/pre_training_neurips/barlowtwins/bt_neurips_pretraining.sh
#SBATCH -t 2-00:00:00
=======
#SBATCH -t 08:00:00
>>>>>>> Stashed changes:scripts/multiomics/pre_training_20m/hvg_gene_program_masking/HVG_MLP_01p_v11.sh
#SBATCH --mem=90GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
python train.py --version 'bt_pretrain_neurips' --model 'BT' --mode 'pre_training' --batch_size 256 --dropout 0.0 --weight_decay 1e-6


