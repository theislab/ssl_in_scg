#!/bin/bash

#SBATCH -J Rec_MLP_Gene_Program_Masking_25p_run3
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000


source $HOME/.profile
conda activate celldreamer
cd ..
cd ..
cd ..
cd ..
cd self_supervision/trainer/reconstruction/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/icb/till.richter/anaconda3/envs/celldreamer/lib
python -u train.py --decoder --lr=0.001 --weight_decay=0.01 --batch_size=16384 --version='run3' --dropout=0.05 --stochastic --pretrained_dir="/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/masking/CN_MLP_gene_program_C8_25p/default/version_2/checkpoints/best_checkpoint_val.ckpt"