#!/bin/bash

# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image
# flux: 1.5/min???, 0.25min/1 image
# kolors: 8.57/min, 0.1166min/1 image
# auraflow: 1.5/min, 0.6667min/1 image

#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
##SBATCH -q big_qos

source ~/.bashrc
ml purge
conda init bash
conda activate generate

IMAGE_DIR='./generated_datasets/hoi_diversified_new_sdxl'
GENERATIVE_MODEL="sdxl" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=3603 # 3603
PROMPT_DIR='../prompt_generation/prompts/hoi_diversified_new.json'

python generate_images_hoi_diversified_new.py -m $GENERATIVE_MODEL -r $IMAGE_DIR -s $START_CLASS -e $END_CLASS -p $PROMPT_DIR

