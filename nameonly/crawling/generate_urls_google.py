import os
import logging
from url_generator.google_generator import GoogleURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

save_dir = './urls/food101_google_clipart'

generator = GoogleURLGenerator(save_dir=save_dir, max_scroll=5, sleep_time=2, mode='default')

# # Check the save directory to remove already generated URLs
# class_txt_files = os.listdir(save_dir)
# class_txt_files = [f.split('.')[0] for f in class_txt_files]
# domainnet = [cls for cls in domainnet if cls not in class_txt_files]

for cls in tqdm(food101_count):
    # if f"{cls}.txt" in os.listdir(save_dir):
    #     logger.info(f"Skipping {cls} because it already exists")
    #     continue
    logger.info(f"Generating URL for {cls}")
    result = generator.generate_url(query=cls, total_images=100, image_type='None')
    breakpoint()