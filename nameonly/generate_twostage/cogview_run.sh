#!/bin/bash
# 10개에 2시간 40분 -> 160분 -> 20개는 320분, 6시간 충분, 40개는 640분, 13시간 충분
# 25개는 

# Floyd: 1분에 2개정도 -> DomainNet 180개 -> 1시간 반, 5개 class 7시간 반

#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
source ~/.bashrc
ml purge

conda init bash
conda activate cogview

cd Image-Local-Attention
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
cd ../
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python get_image_onestage.py --config_path ./configs/PACS_cogview2.yaml --start_class 0 --end_class 1
