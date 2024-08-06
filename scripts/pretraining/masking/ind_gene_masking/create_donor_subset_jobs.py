import os

# Directory containing donor list files
donor_dir = "/lustre/groups/ml01/workspace/till.richter/scTab"
donor_files = [f for f in os.listdir(donor_dir) if f.startswith("donor_id_subset_") and f.endswith(".npy")]

# SLURM script template for donor files
donor_slurm_template = """#!/bin/bash

#SBATCH -J Pretrain_Data_{version}
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t {time}
#SBATCH --mem=150GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

python -u train.py --mask_rate 0.5 --model 'MLP' --dropout 0.1 --weight_decay 0.01 --lr 0.001 --decoder --version 'data_{version}' --donor_list {donor_list}
"""

# Function to determine walltime based on subset value
def get_walltime(subset):
    if float(subset) < 0.1:
        return "12:00:00"
    else:
        return "1-00:00:00"

# Generate and submit SLURM scripts for donor files
for donor_file in donor_files:
    # Extract subset and seed from filename
    parts = donor_file.split('_')
    subset = parts[3]
    seed = parts[5].split('.')[0]
    
    # Define version and walltime
    version = f"{subset}_seed_{seed}"
    walltime = get_walltime(subset)
    
    # Create SLURM script content
    slurm_content = donor_slurm_template.format(
        version=version,
        time=walltime,
        donor_list=os.path.join(donor_dir, donor_file)
    )
    
    # Write SLURM script to file
    slurm_filename = f"slurm_pretrain_{version}.sh"
    with open(slurm_filename, 'w') as slurm_file:
        slurm_file.write(slurm_content)
    
    # Submit the SLURM script
    os.system(f"sbatch {slurm_filename}")
