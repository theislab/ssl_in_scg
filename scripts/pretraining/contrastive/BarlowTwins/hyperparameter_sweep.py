import os
import itertools
import subprocess

# Define the parameter grid
param_grid = {
    "p": [0.5, 0.7],
    "negbin_intensity": [0.2, 0.4],
    "dropout_intensity": [0.2],
}

combinations = list(itertools.product(*param_grid.values()))

print('Parameter combinations:', len(combinations))

# Load the job script template
with open('job_template.sh', 'r') as file:
    template = file.read()

# Check for NULL characters in the template
if '\0' in template:
    raise ValueError("The job template contains NULL characters, which are not allowed.")

os.makedirs('slurm_jobs', exist_ok=True)

for index, params in enumerate(combinations):
    negbin_intensity, dropout_intensity, p = params
    script_content = template.format(
        index=index,
        negbin_intensity=negbin_intensity,
        dropout_intensity=dropout_intensity,
        p=p
    )
    script_path = f'slurm_jobs/job_{index}.sh'

    # Check for NULL characters in the script content
    if '\0' in script_content:
        raise ValueError(f"The generated script {script_path} contains NULL characters, which are not allowed.")

    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

    # Submit the job
    subprocess.run(['sbatch', script_path])
