import os
import json
import logging
from url_generator.google_generator import GoogleURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

save_dir = './urls/bongard_openworld'
error_file_path = './openworld_pos_error.txt'

generator = GoogleURLGenerator(max_scroll=1, sleep_time=2, mode='headless')

# # Check the save directory to remove already generated URLs
# class_txt_files = os.listdir(save_dir)
# class_txt_files = [f.split('.')[0] for f in class_txt_files]
# domainnet = [cls for cls in domainnet if cls not in class_txt_files]

# # For classification
# for cls in tqdm(food101_count):
#     logger.info(f"Generating URL for {cls}")
#     url_list = generator.generate_url(query=cls, total_images=10, image_type='None')
#     with open(os.path.join(save_dir, f'{cls}.txt'), 'w') as f:
#         for url in url_list:
#             f.write(url + '\n')

# For Bongard-Openworld - positive 7, negative 7
# Process jsonl file
data_list = []
with open('../prompt_generation/prompts/openworld_base.json', 'r') as f:
    data_dict = json.load(f)

# Crawl positive images
generator = GoogleURLGenerator(max_scroll=5, sleep_time=2, mode='headless')
for uid, pos_neg_dict in tqdm(data_dict.items()):
    pos_save_dir = os.path.join(save_dir, 'pos')
    txt_file_path = os.path.join(pos_save_dir, f"{uid}.txt")

    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 10:
            logger.info(f"Skipping already processed uid - {uid}")
            continue
        else:
            logger.info(f"uid - {uid} file exists, but not enough ({len(lines)}) urls!")
    
    logger.info(f"Generating URLs for uid - {uid}")
    pos_url_list = generator.generate_url(query=pos_neg_dict['positive_prompts'][0], total_images=100, image_type='None')
    if len(pos_url_list) < 10:
        with open(error_file_path, 'a') as error_file:
            error_file.write(f"uid: {uid}, List Length: {len(pos_url_list)}\n")
    pos_save_dir = os.path.join(save_dir, 'pos')
    os.makedirs(pos_save_dir, exist_ok=True)
    with open(os.path.join(pos_save_dir, f"{uid}.txt"), 'w') as f:
        for url in pos_url_list:
            f.write(url + '\n')

# # Crawl positive images
# generator = GoogleURLGenerator(max_scroll=1, sleep_time=2, mode='headless')
# for uid, pos_neg_dict in tqdm(data_dict.items()):
#     neg_save_dir = os.path.join(save_dir, 'neg')
#     txt_file_path = os.path.join(neg_save_dir, f"{uid}.txt")

#     if os.path.exists(txt_file_path):
#         with open(txt_file_path, 'r') as f:
#             lines = f.readlines()
#         if len(lines) > 10:
#             logger.info(f"Skipping already processed uid - {uid}")
#             continue
#         else:
#             logger.info(f"uid - {uid} file exists, but not enough ({len(lines)}) urls!")
    
#     logger.info(f"Generating URLs for uid - {uid}")
#     neg_url_list = []
#     negative_prompts = pos_neg_dict['negative_prompts']
#     for neg_prompt in negative_prompts:
#         try:
#             neg_urls = generator.generate_url(query=neg_prompt, total_images=15, image_type='None')
#         except Exception as e:
#             print(f"Error occured while processing uid - {uid} - {e}")
#         if len(neg_urls) > 15:
#             neg_urls = neg_urls[:15]
#         neg_url_list.extend(neg_urls)

#     if len(neg_url_list) < 10:
#         with open(error_file_path, 'a') as error_file:
#             error_file.write(f"uid: {uid}, List Length: {len(neg_url_list)}\n")
#     os.makedirs(neg_save_dir, exist_ok=True)
#     with open(os.path.join(neg_save_dir, f"{uid}.txt"), 'w') as f:
#         for url in neg_url_list:
#             f.write(url + '\n')
