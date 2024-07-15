import os
import subprocess

# Define the parameters
supervised_subsets = ["HLCA", "PBMC", "Tabula_Sapiens"]
pretrained_models = [
    ### 56.2% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.5623413251903491_seed_1/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.5623413251903491_seed_2/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.5623413251903491_seed_3/default/version_1/checkpoints/best_checkpoint_val.ckpt",
    ### 31.6% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.31622776601683794_seed_1/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.31622776601683794_seed_2/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.31622776601683794_seed_3/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    ### 17.8% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.1778279410038923_seed_1/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.1778279410038923_seed_2/default/version_2/checkpoints/best_checkpoint_val.ckpt", 
    # "pretext_models/masking/CN_MLP_50pdata_0.1778279410038923_seed_3/default/version_1/checkpoints/best_checkpoint_val.ckpt", 
    ### 10.0% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.1_seed_1/default/version_3/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.1_seed_2/default/version_1/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.1_seed_3/default/version_1/checkpoints/best_checkpoint_val.ckpt",
    ### 3.16% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.03162277660168379_seed_1/default/version_1/checkpoints/best_checkpoint_val.ckpt", 
    # "pretext_models/masking/CN_MLP_50pdata_0.03162277660168379_seed_2/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # "pretext_models/masking/CN_MLP_50pdata_0.03162277660168379_seed_3/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    ### 1.00% ###
    # "pretext_models/masking/CN_MLP_50pdata_0.01_seed_1/default/version_2/checkpoints/best_checkpoint_val.ckpt",
    # 'pretext_models/masking/CN_MLP_50pdata_0.01_seed_2/default/version_2/checkpoints/best_checkpoint_val.ckpt',
    # "pretext_models/masking/CN_MLP_50pdata_0.01_seed_3/default/version_2/checkpoints/best_checkpoint_val.ckpt",
]

# Load the job script template
with open('classifier_template.sh', 'r') as file:
    template = file.read()

os.makedirs('classifier_jobs', exist_ok=True)
print('Generating job scripts for {} combinations'.format(len(pretrained_models) * len(supervised_subsets)))

# Generate job scripts
for pretrained_model in pretrained_models:
    pretrained_model_name = pretrained_model.split('/')[2]
    
    for supervised_subset in supervised_subsets:
        version = f"donor_{pretrained_model_name}_{supervised_subset}"
        script_content = template.format(
            supervised_subset=supervised_subset,
            pretrained_model_name=pretrained_model_name,
            version=version,
            pretrained_dir=pretrained_model
        )
        script_path = f'classifier_jobs/job_{version}.sh'
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        
        # Submit the job
        subprocess.run(['sbatch', script_path])
