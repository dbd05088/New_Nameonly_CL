#!/bin/bash

# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image
# flux: 1.5/min???, 0.25min/1 image
# kolors: 8.57/min, 0.1166min/1 image
# auraflow: 1.5/min, 0.6667min/1 image

#SBATCH -p suma_a6000
##SBATCH -q big_qos
#SBATCH --gres=gpu:1
##SBATCH --exclude node37

source ~/.bashrc
ml purge
conda init bash
conda activate cogview

cd Image-Local-Attention
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
cd ../
IMAGE_DIR='./generated_datasets/generated_LLM_cogview2_ver3'
GENERATIVE_MODEL="cogview2" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=3603 # 3603
PROMPT_DIR='../prompt_generation/prompts/generated_LLM_sdxl_ver3.json'

CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python generate_images_hoi_diversified_new.py -m $GENERATIVE_MODEL -r $IMAGE_DIR -s $START_CLASS -e $END_CLASS -p $PROMPT_DIR

# Non-slurm cluster: python generate_images_hoi_diversified_new.py -m cogview2 -r ./generated_datasets/generated_LLM_cogview2_ver3 -s 3400 -e 3603 -p ../prompt_generation/prompts/generated_LLM_sdxl_ver3.json