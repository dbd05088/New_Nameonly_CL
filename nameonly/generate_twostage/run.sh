#!/bin/bash
# 10개에 2시간 40분 -> 160분 -> 20개는 320분, 6시간 충분, 40개는 640분, 13시간 충분
# 25개는 

# Floyd - 4090: 1분에 2.2857개 / gpu, 0.4375분 / 1개
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1

source ~/.bashrc
ml purge

conda init bash
conda activate generate
python get_image_onestage.py --config_path ./configs/kandinsky2.yaml --start_class 0 --end_class 0
