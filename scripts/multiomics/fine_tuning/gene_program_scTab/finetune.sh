#!/bin/bash

#SBATCH -J FineTune_GeneProgram_20M_run
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 1-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000
#SBATCH --distribution=block:block

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
python train.py --mode 'fine_tuning' --version 'run0' --dropout 0.09673756767199265 --weight_decay 0.0012139742287047533 --learning_rate 0.0007534830977174946 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gene_program_MAE_GP_Pretrain_20M_v2/default/version_0/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run1' --dropout 0.09673756767199265 --weight_decay 0.0012139742287047533 --learning_rate 0.0007534830977174946 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gene_program_MAE_GP_Pretrain_20M_v2/default/version_0/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run2' --dropout 0.09673756767199265 --weight_decay 0.0012139742287047533 --learning_rate 0.0007534830977174946 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gene_program_MAE_GP_Pretrain_20M_v2/default/version_0/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run3' --dropout 0.09673756767199265 --weight_decay 0.0012139742287047533 --learning_rate 0.0007534830977174946 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gene_program_MAE_GP_Pretrain_20M_v2/default/version_0/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run4' --dropout 0.09673756767199265 --weight_decay 0.0012139742287047533 --learning_rate 0.0007534830977174946 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gene_program_MAE_GP_Pretrain_20M_v2/default/version_0/checkpoints/best_checkpoint_val.ckpt

