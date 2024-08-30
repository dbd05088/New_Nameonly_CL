#!/bin/bash
# 10개에 2시간 40분 -> 160분 -> 20개는 320분, 6시간 충분, 40개는 640분, 13시간 충분
# 25개는 

# Floyd: 1분에 2개정도 -> DomainNet 180개 -> 1시간 반, 5개 class 7시간 반
# Cogview2: 72분에 class 1개, 1분에 3.57개, initialize는 7분정도 걸림.
# 1개에 0.2784분 -> 1개 class에 30분 정도 걸린다고 보면 된다. 40개 class정도 생성하면 될듯.

#SBATCH -p suma_a6000
#SBATCH --gres=gpu:1
source ~/.bashrc
ml purge

conda init bash
conda activate cogview

cd Image-Local-Attention
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
cd ../

DATASET="DomainNet" # PACS, DomainNet, cifar10, NICO
IMAGE_DIR='./generated_datasets/DomainNet_cogview2'
GENERATIVE_MODEL="cogview2" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=0
PROMPT_DIR='../prompt_generation/prompts/gpt4_hierarchy_cot_1.json'
INCREASE_RATIO=1.2

CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python get_image_onestage.py --config_path ./configs/default.yaml --dataset $DATASET --image_dir $IMAGE_DIR --generative_model $GENERATIVE_MODEL --start_class $START_CLASS --end_class $END_CLASS --prompt_dir $PROMPT_DIR --increase_ratio $INCREASE_RATIO
