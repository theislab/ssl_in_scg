#!/bin/bash

#SBATCH -J rec_cn_byol_pre
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


# sbatch '01_run0.sh' &
# sbatch '001_run0.sh' &
# sbatch '0001_run0.sh' &
sbatch '01_run1.sh' &
sbatch '001_run1.sh' &
sbatch '0001_run1.sh' &
sbatch '01_run2.sh' &
sbatch '001_run2.sh' &
sbatch '0001_run2.sh' &
sbatch '01_run3.sh' &
sbatch '001_run3.sh' &
sbatch '0001_run3.sh' &
sbatch '01_run4.sh' &
sbatch '001_run4.sh' &
sbatch '0001_run4.sh' &
wait