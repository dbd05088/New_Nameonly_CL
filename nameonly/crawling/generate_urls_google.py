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
    url_list = generator.generate_url(query=cls, total_images=10, image_type='None')
    breakpoint()    
    
    with open(os.path.join(save_dir, f'{filename}.txt'), 'w') as f:
        for url in self.url_list:
            f.write(url + '\n')

# For Bongard-Openworld - positive 7, negative 7
# Process jsonl file
data_list = []
with open('../prompt_generation/prompts/openworld_base.json', 'r') as f:
    data_dict = json.load(f)

for uid, pos_neg_dict in data_dict.keys():
    logger.info(f"Generating URLs for uid - {uid}")
    