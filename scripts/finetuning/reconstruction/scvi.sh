#!/bin/bash

#SBATCH -J scvi_cellxgene
#SBATCH -p cpu_p
#SBATCH -t 2-00:00:00
#SBATCH --mem=700GB
#SBATCH --cpus-per-task=48
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi
cd $SSL_PROJECT_HOME/self_supervision/trainer/reconstruction/

python -u train_scvi.py