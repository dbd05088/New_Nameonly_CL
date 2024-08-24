#!/bin/bash

# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image

#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1

source ~/.bashrc
ml purge
conda init bash
conda activate generate

DATASET="PACS" # PACS, DomainNet, cifar10, NICO
IMAGE_DIR='./generated_datasets/PACS_sdxl'
GENERATIVE_MODEL="sdxl" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=0
PROMPT_DIR='../prompt_generation/prompts/gpt4_hierarchy_cot_1.json'
INCREASE_RATIO=1.2

python get_image_onestage.py --config_path ./configs/default.yaml --dataset $DATASET --image_dir $IMAGE_DIR --generative_model $GENERATIVE_MODEL --start_class $START_CLASS --end_class $END_CLASS --prompt_dir $PROMPT_DIR --increase_ratio $INCREASE_RATIO
