#!/bin/bash

#SBATCH -J hvg_selection
#SBATCH -p cpu_p
#SBATCH --qos cpu_long
#SBATCH -t 2-00:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

python -u hvg_selection.py