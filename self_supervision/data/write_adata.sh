#!/bin/bash

#SBATCH -J write_cellxgene_adata
#SBATCH -p cpu_p
#SBATCH -q cpu_long
#SBATCH -t 2-00:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=20
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi


splits=("train" "val" "test")
percs=(100) #(100 90 80 70 60 50 40 30 20 10)

for split in "${splits[@]}"
do
    for perc in "${percs[@]}"
    do
        echo "Running write_adata.py with split: $split and perc: $perc..."
        python write_adata.py --perc $perc --split $split --adata_dir /lustre/groups/ml01/workspace/$USER/cellxgene/
        echo "Done."
    done
done