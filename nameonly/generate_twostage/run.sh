#!/bin/bash

source ~/.bashrc
ml purge

conda init bash
conda activate generate
python get_image_onestage.py --config_path ./configs/default.yaml --start_class 0 --end_class 9 
