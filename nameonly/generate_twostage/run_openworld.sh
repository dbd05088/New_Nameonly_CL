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

# OpenWorld Range: 0 ~ 609
python generate_images_openworld.py -m sdturbo -r ./generated_datasets/openworld_diversified_sdturbo -s 305 -e 609
