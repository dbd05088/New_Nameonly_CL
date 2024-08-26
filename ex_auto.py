import os

# Script file name
script_file = "ex_auto.sh"


# Set variables
MODE = "der"
MODEL_NAME = "resnet18"
DATASET = "PACS_final"
TYPES = [
    "sdxl_floyd_cogview2_sd3_auraflow_equalweight",
    # "sdxl_floyd_cogview2_sd3_equalweight",
    # "sdxl_floyd_cogview2_sd3_flux_auraflow_equalweight",
    # "sdxl_floyd_cogview2_sd3_flux_equalweight",
    # "sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_equalweight",
    # "sdxl_floyd_cogview2_sd3_flux_kolors_equalweight",
    # "sdxl_floyd_cogview2_sd3_kolors_auraflow_equalweight",
]

# Generate sbatch commands and submit jobs
for seed in ['1', '2', '3', '4', '5']:
    for type_name in TYPES:
        sbatch_command = f"sbatch --export=ALL,MODE={MODE},MODEL_NAME={MODEL_NAME},DATASET={DATASET},TYPE={type_name},SEED={seed} {script_file}"
        print(f"Submitting job with MODE={MODE}, MODEL_NAME={MODEL_NAME}, DATASET={DATASET}, TYPE={type_name}, SEED={seed}")
        os.system(sbatch_command)
