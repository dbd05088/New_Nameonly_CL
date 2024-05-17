#!/bin/bash

source ~/.bashrc
ml purge

conda init bash
conda activate generate
python get_image_onestage.py --config_path ./configs/default2.yaml
