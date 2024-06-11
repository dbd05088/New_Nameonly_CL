import os
import random
import shutil
import argparse
from tqdm import tqdm
from classes import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-s', '--source_path', type=str, required=True)
parser.add_argument('-t', '--target_path', type=str, required=True)
parser.add_argument('-r', '--subsample_ratio', type=float)
parser.add_argument('-n', '--subsample_count', type=int)

args = parser.parse_args()

dataset_mapping = {'DomainNet': DomainNet_count, 'officehome': officehome_count, 'PACS': PACS_count, 
                   'birdsnap': birdsnap_count, 'cifar10': cifar10_count, 'aircraft': aircraft_count,
                   'food101': food101_count, 'cct': cct_count, 'pacs_sdxl': pacs_sdxl_count, 
                   'pacs_dalle2': pacs_dalle2_count, 'pacs_deepfloyd': pacs_deepfloyd_count,
                   'pacs_cogview2': pacs_cogview2_count, 'pacs_sdxl_new': pacs_sdxl_new_count,
                   'pacs_dalle2_new': pacs_dalle2_new_count, 'NICO': NICO_count}

# CHANGE THIS!!!
source_path = args.source_path
target_path = args.target_path
count_dict = dataset_mapping[args.dataset]
subsample_ratio = args.subsample_ratio
subsample_count = args.subsample_count

classes = count_dict.keys()

for cls in tqdm(classes):
    if subsample_ratio is not None:
        subsample_num = int(count_dict[cls] * subsample_ratio)
    elif subsample_count is not None:
        subsample_num = subsample_count

    source_class_path = os.path.join(source_path, cls)
    target_class_path = os.path.join(target_path, cls)

    source_image_paths = os.listdir(source_class_path)
    random.shuffle(source_image_paths)

    if not os.path.exists(target_class_path):
        os.makedirs(target_class_path)

    # Choose the first subsample_num images
    try:
        for i in range(subsample_num):
            source_image = source_image_paths[i]
            source_image_path = os.path.join(source_class_path, source_image)
            target_image_path = os.path.join(target_class_path, source_image)
            shutil.copy(source_image_path, target_image_path)
    except Exception as e:
        print(f"While processing class {cls}, error occured...")
        print(f"Length from cls dict: {subsample_num}")
        print(f"Length of dir: {len(source_image_paths)}")
        print(e)
        continue

