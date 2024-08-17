#!/bin/bash

# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image

#SBATCH -p suma_a6000
##SBATCH -q big_qos
#SBATCH --gres=gpu:1
source ~/.bashrc
ml purge

conda init bash
conda activate cogview

cd Image-Local-Attention
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
cd ../
# HOI Range: 0 ~ 227
# 207, 165, 124, 151, 225 indices contain many images
# Average number of images per class: 17 ~ 40
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python generate_images_hoi.py -m cogview2 -r ./generated_datasets/hoi_cogview2 -s 0 -e 28



