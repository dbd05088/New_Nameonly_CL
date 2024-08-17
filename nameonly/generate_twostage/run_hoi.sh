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
python generate_images_hoi.py -m floyd -r ./generated_datasets/hoi_floyd -s 210 -e 227
