#!/usr/bin/env python3

import cv2
import os
from classes import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-s', '--source_path', type=str, required=True)
parser.add_argument('-r', '--threshold_ratio', type=float)
args = parser.parse_args()

def check_class_names(directory_path, class_names_dict):
    class_names = os.listdir(directory_path)
    class_names = [cls for cls in class_names if not cls.endswith('.json')]
    for cls in class_names:
        if cls not in class_names_dict.keys():
            print(f"Class {cls} not found in class names dictionary.")
            raise ValueError(f"Class {cls} not found in class names dictionary.")
    print(set(class_names_dict.keys()) - set(class_names))
    assert len(class_names) == len(class_names_dict.keys()), "Number of classes in directory does not match number of classes in class names dictionary."
    
def check_image_size(image_path, size):
    image = cv2.imread(image_path)
    if image.shape[:2] != size:
        print(f"Image {image_path} has shape {image.shape} which does not match the expected shape {size}")
        return False
    return True

def convert_images_in_directory(directory_path):
    print(f"Checking corrupted images in {directory_path}")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    corrupted_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                # print(f"Processing {file_path}")
                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        raise ValueError("Image could not be read, possibly due to a format issue or file corruption.")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    corrupted_files.append(file_path)

    print(f"Corrupted files: {corrupted_files}")

    if len(corrupted_files) == 0:
        return True
    else:
        return False

# CHANGE THIS!!!
directory_path = args.source_path


dataset_mapping = {'DomainNet': DomainNet_count, 'officehome': officehome_count, 'PACS': PACS_count, 
                   'birdsnap': birdsnap_count, 'cifar10': cifar10_count, 'aircraft': aircraft_count,
                   'food101': food101_count, 'cct': cct_count, 'pacs_sdxl': pacs_sdxl_count, 
                   'pacs_dalle2': pacs_dalle2_count, 'pacs_deepfloyd': pacs_deepfloyd_count,
                   'pacs_cogview2': pacs_cogview2_count, 'pacs_sdxl_new': pacs_sdxl_new_count,
                   'pacs_dalle2_new': pacs_dalle2_new_count, 'NICO': NICO_count}

class_names_dict = dataset_mapping[args.dataset]

cls_count_dict = {cls: len(os.listdir(os.path.join(directory_path, cls))) for cls in os.listdir(directory_path) if not cls.endswith('.json')}
print(f"Total number of classes: {len(os.listdir(directory_path))}")
print(f"Dictionary from classes.py: {class_names_dict}")
print(f"Count dictionary from dir: {cls_count_dict}")
print(f"Size same: {class_names_dict == cls_count_dict}")
print(f"Key same: {class_names_dict.keys() == cls_count_dict.keys()}")
check_class_names(directory_path, class_names_dict)

# Choose all images in the first class
class_names = [cls for cls in os.listdir(directory_path) if not cls.endswith('.json')]
class_name = class_names[0]
for image in os.listdir(os.path.join(directory_path, class_name)):
    if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png'):
        # Check
        image_path = os.path.join(directory_path, class_name, image)
        # print(f"Checking image size...")
        image_size_correct = check_image_size(image_path, (224, 224))
        if not image_size_correct:
            print(f"WARNING: Image {image_path} does not have the correct size.")
            break
print(f"Image size correct: {image_size_correct}")

convert_correct = convert_images_in_directory(directory_path)

if image_size_correct and convert_correct and {class_names_dict.keys() == cls_count_dict.keys()}:
    print(f"SUCCESS: All images in {directory_path} are valid.")
else:
    print(f"!!!!!WARNING: Some images in {directory_path} are invalid.")

