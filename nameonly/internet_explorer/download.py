import os
import sys
import json
import shutil
from tqdm import tqdm
from better_bing_image_downloader import downloader

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(target_dir)
from classes import get_count_dict

count_dict = get_count_dict("DomainNet")
descriptors = "DomainNet_descriptors.json"
target_dir = "DomainNet_internet_explorer"
start_index = 0; end_index = 344
increase_ratio = 1.15
concepts = list(count_dict.keys())
concepts = concepts[start_index:end_index + 1]


image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']

with open(descriptors, 'r') as f:
    descriptors_dict = json.load(f)
    

for concept in tqdm(concepts):
    download_path = os.path.join(target_dir, concept)
    current_count = 0
    min_images = int(count_dict[concept] * increase_ratio)
    if os.path.exists(download_path) and len(os.listdir(download_path)) >= min_images:
        print(f"Skipping {concept} as it already has {min_images} images")
        continue
    
    print(f"Downloading images for {concept} with a minimum of {min_images} images")
    descriptors_list = descriptors_dict[concept]
    
    for i, descriptor in enumerate(descriptors_list):
        if current_count >= min_images:
            break
        query_string = f"{descriptor} {concept}"
        downloader(query_string, limit=20, output_dir=download_path, adult_filter_off=True, 
                   timeout=60, filter="", verbose=True, badsites= [], name=f"Image_{i}")
        
        # Count the number of images recursively
        current_count = sum([len(files) for root, dirs, files in os.walk(download_path)])
        print(f"Current count for {concept}: {current_count} / {min_images}")

    print(f"Finished downloading images for {concept}")
    
    # Reorganize the image directory
    image_num = 0
    dirs = [d for d in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, d))]
    # Move all images to the root directory
    for d in dirs:
        images = [f for f in os.listdir(os.path.join(download_path, d)) if os.path.isfile(os.path.join(download_path, d, f))]
        images = [f for f in images if os.path.splitext(f)[1] in image_exts]
        for image in images:
            shutil.copy(os.path.join(download_path, d, image), os.path.join(download_path, f"{str(image_num).zfill(5)}{os.path.splitext(image)[1]}"))
            image_num += 1
    
    # Remove all subdirectories
    for d in dirs:
        shutil.rmtree(os.path.join(download_path, d))