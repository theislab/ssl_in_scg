#!/bin/bash

#SBATCH -J Multi_Finetune_GP_to_TF_20M
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
python train.py --mode 'fine_tuning' --version 'run0' --dropout 0.005949689231703099 --weight_decay 0.0010722296291834556 --learning_rate 0.0009379974005232709 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gp_to_tf_MAE_v1/default/version_4/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run1' --dropout 0.005949689231703099 --weight_decay 0.0010722296291834556 --learning_rate 0.0009379974005232709 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gp_to_tf_MAE_v1/default/version_4/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run2' --dropout 0.005949689231703099 --weight_decay 0.0010722296291834556 --learning_rate 0.0009379974005232709 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gp_to_tf_MAE_v1/default/version_4/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run3' --dropout 0.005949689231703099 --weight_decay 0.0010722296291834556 --learning_rate 0.0009379974005232709 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gp_to_tf_MAE_v1/default/version_4/checkpoints/best_checkpoint_val.ckpt
python train.py --mode 'fine_tuning' --version 'run4' --dropout 0.005949689231703099 --weight_decay 0.0010722296291834556 --learning_rate 0.0009379974005232709 --pretrained_dir /lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/multiomics/multiomics_20M_gp_to_tf_MAE_v1/default/version_4/checkpoints/best_checkpoint_val.ckpt
