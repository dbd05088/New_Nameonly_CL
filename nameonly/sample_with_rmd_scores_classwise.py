# 이제는 equalweight용
import os, random, shutil
import pickle
import numpy as np
import json
from classes import *
from utils import softmax_with_temperature
from tqdm import tqdm
from pathlib import Path

Equalweight = True
TopK = False
BottomK = False

INVERSE = False
TEMPERATURE = 2

# NICO
base_path = './raw_datasets/iclr_generated/PACS'
json_path = './RMD_scores/PACS_final_sd3.json'
target_path = './raw_datasets/iclr_generated/PACS/PACS_sd3_equalweight'

count_dict = get_count_value_from_string(base_path)
with open(json_path, 'r') as f:
    RMD_scores = json.load(f)

# Parse RMD pickle file to get PATH dict
models = list(RMD_scores.keys())
PATH_dict = {}
for model in models:
    first_class = next(iter(RMD_scores[model]))
    relative_path = RMD_scores[model][first_class][0]['image_path'] # Get the first item
    PATH_dict[model] = os.path.join(base_path, str(Path(relative_path).parents[1]))

# Shuffle the images
for model in PATH_dict.keys():
    for cls in count_dict.keys():
        random.shuffle(RMD_scores[model][cls])

ensembled_images = {cls: [] for cls in count_dict.keys()}
model_class_selected_counter = {model: {cls: 0 for cls in count_dict.keys()} for model in PATH_dict.keys()}

for cls in tqdm(list(count_dict.keys())):
    # First convert the mapping
    # Original(RMD_scores): {(model, cls): [(image_path, RMD_score), ...]}
    # New: {model: RMD_score, ...}

    model_RMD_mapping = {model: [] for model in PATH_dict.keys()} # Model to RMD scores list of current class
    for (model, cls_), images in RMD_scores.items():
        if cls_ == cls:
            for image, score in images:
                model_RMD_mapping[model].append(score)
    
    # Get average RMD score for each model (in-place)
    for model, scores in model_RMD_mapping.items():
        model_RMD_mapping[model] = np.mean(scores)
        print(f"Class {cls}, model {model}, average RMD score: {model_RMD_mapping[model]}, length: {len(scores)}")
    
    if 'dalle2' in model_RMD_mapping and cls == 'underwear':
        model_RMD_mapping['dalle2'] = 0.01 # To avoid numerical instability
    
    # Randomly choose the model according to the ensemble probability
    while True:
        scores = np.array([score for score in model_RMD_mapping.values()])
        probabilities = softmax_with_temperature(scores, TEMPERATURE)
        if Equalweight:
            probabilities = [1 / len(model_RMD_mapping)] * len(model_RMD_mapping)
        
        if INVERSE:
            # To get the inverse probabilities, first handle the numerical instability
            if np.min(probabilities) < 0:
                probabilities -= np.min(probabilities)
            # Handle devision by zero
            if np.sum(probabilities) == 0:
                raise ValueError("All probabilities are zero")
            probabilities = 1 / probabilities
            probabilities /= np.sum(probabilities) # Normalize the probabilities
        
        chosen_model = random.choices(list(model_RMD_mapping.keys()), weights=probabilities, k=1)[0]
        if len(model_to_images_dict[chosen_model][cls]) > 0:
            model_class_selected_counter[chosen_model][cls] += 1
            chosen_image = model_to_images_dict[chosen_model][cls].pop()
            chosen_image_path = os.path.join(PATH_dict[chosen_model], cls, chosen_image)
            ensembled_images[cls].append({'model': chosen_model, 'image': chosen_image_path})
        if len(ensembled_images[cls]) == count_dict[cls]:
            print(f"Break for class {cls} with {len(ensembled_images[cls])} images")
            break

# Check the number of images selected for each model, for each class
for model, class_counter in model_class_selected_counter.items():
    print(f"Model {model} selected for each class:")
    print(class_counter)

# Sanity check the number of images for each class
for cls, images in ensembled_images.items():
    if len(images) != count_dict[cls]:
        raise ValueError(f"Class {cls} has {count_dict[cls]} images but {len(images)} images are selected")

# Copy all the images to the target path
# Remove target path if already exists
if os.path.exists(target_path):
    print(f"[WARNING]: Removing already existing target path - {target_path}")
    shutil.rmtree(target_path)

for cls, images in ensembled_images.items():
    image_counter = 0
    target_cls_path = os.path.join(target_path, cls)
    os.makedirs(target_cls_path, exist_ok=True)
    for img in images:
        model = img['model']; image_name = img['image']
        new_image_name = f"{model}_{str(image_counter).zfill(6)}.png"
        image_counter += 1
        shutil.copy(image_name, os.path.join(target_cls_path, new_image_name))