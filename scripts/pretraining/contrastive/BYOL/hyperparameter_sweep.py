import os
import itertools
import subprocess

# Define the parameter grid
param_grid = {
    "p": [0.1, 0.3],
    "negbin_intensity": [0.7, 0.9],
    "dropout_intensity": [0.7, 0.9],
}

combinations = list(itertools.product(*param_grid.values()))

print('Parameter combinations:', len(combinations))

# Remove combinations already explored
model_path = "/lustre/groups/ml01/workspace/till.richter/trained_models/pretext_models/contrastive"

# models are stored in subdirectories with folder names containing, e.g., "BYOL_p_0_3_negbin_0_2_dropout_0_05" showing p=0.3, negbin_intensity=0.2, dropout_intensity=0.05
# define dummy model names from parameter combination and check if they exist in the model_path
# if they exist, remove the combination from the list of combinations
dummy_model_names = [f"BYOL_p_{int(p*10)}_negbin_{int(negbin_intensity*10)}_dropout_{int(dropout_intensity*10)}" for p, negbin_intensity, dropout_intensity in combinations]
# check if the dummy model names is a substring of any folder name in the model_path
for model_name in os.listdir(model_path):
    for dummy_model_name in dummy_model_names:
        if dummy_model_name in model_name:
            print(f"Removing {dummy_model_name} from the list of combinations")
            dummy_model_names.remove(dummy_model_name)

print('New parameter combinations:', len(dummy_model_names))
# update combinations
combinations = [(p, negbin_intensity, dropout_intensity) for p, negbin_intensity, dropout_intensity in combinations if f"BYOL_p_{int(p*10)}_negbin_{int(negbin_intensity*10)}_dropout_{int(dropout_intensity*10)}" in dummy_model_names]

# Load the job script template
with open('job_template.sh', 'r') as file:
    template = file.read()

os.makedirs('slurm_jobs', exist_ok=True)

for index, params in enumerate(combinations):
    p, negbin_intensity, dropout_intensity= params
    script_content = template.format(
        index=index,
        p=p,
        negbin_intensity=negbin_intensity,
        dropout_intensity=dropout_intensity,
    )
    script_path = f'slurm_jobs/job_{index}.sh'
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

    # Submit the job
    subprocess.run(['sbatch', script_path])
