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

# HOI Range: 0 ~ 227
# 207, 165, 124, 151, 225 indices contain many images
# Average number of images per class: 17 ~ 40
IMAGE_DIR='./generated_datasets/hoi_diversified_sdxl'
GENERATIVE_MODEL="sdxl" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=227
PROMPT_DIR='../prompt_generation/prompts/hoi_diversified.json'

python generate_images_hoi_diversified.py -m $GENERATIVE_MODEL -r $IMAGE_DIR -s $START_CLASS -e $END_CLASS -p $PROMPT_DIR

