import os
import json
import logging
from url_generator.google_generator import GoogleURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

save_dir = './urls/bongard_openworld'

generator = GoogleURLGenerator(save_dir=save_dir, max_scroll=5, sleep_time=2, mode='headless')

# # Check the save directory to remove already generated URLs
# class_txt_files = os.listdir(save_dir)
# class_txt_files = [f.split('.')[0] for f in class_txt_files]
# domainnet = [cls for cls in domainnet if cls not in class_txt_files]

# For classification
for cls in tqdm(food101_count):
    logger.info(f"Generating URL for {cls}")
    result = generator.generate_url(query=cls, total_images=100, image_type='None')

# For Bongard-Openworld - positive 7, negative 7
# Process jsonl file
data_list = []
with open('../generate_twostage/train.jsonl', 'r') as f:
    for line in f:
        json_object = json.loads(line)
        data_list.append(json_object)
