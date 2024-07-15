import os
import itertools
import subprocess

# Define the parameter grid
param_grid = {
    "p": [0.7, 0.9],
    "negbin_intensity": [0.7, 0.9],
    "dropout_intensity": [0.7, 0.9],
}

combinations = list(itertools.product(*param_grid.values()))

print('Parameter combinations:', len(combinations))

# Load the job script template
with open('job_template.sh', 'r') as file:
    template = file.read()

os.makedirs('slurm_jobs', exist_ok=True)

for index, params in enumerate(combinations):
    p, negbin_intensity, dropout_intensity = params
    script_content = template.format(
        index=index,
        negbin_intensity=negbin_intensity,
        dropout_intensity=dropout_intensity,
        p=p
    )
    script_path = f'slurm_jobs/job_{index}.sh'
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

    # Submit the job
    subprocess.run(['sbatch', script_path])
