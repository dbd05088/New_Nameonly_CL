import os, random, shutil
import pickle
import numpy as np
from classes import *
from tqdm import tqdm
from pathlib import Path

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

# NORMALIZATION, CLIPPING
normalize = True
clip = True
lower_percentile = 5.0 # 5.0
# lower_percentile = 2.5 # 5.0
upper_percentile = 95.0 # 95.0
# upper_percentile = 97.5 # 95.0
TopK = False
BottomK = False

INVERSE = False
TEMPERATURE = 0.5

count_dict = PACS_count
rmd_pickle_path = './RMD_scores/PACS_final_sd3.pkl'
target_path = './raw_datasets/iclr_generated/PACS/PACS_sd3_RMD'

# Load the RMD scores
with open(rmd_pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

# Parse RMD pickle file to get PATH dict
models = list(set([key[0] for key in RMD_scores.keys()]))
PATH_dict = {}
for (model, cls), score_list in RMD_scores.items():
    if model in PATH_dict:
        continue
    else:
        PATH_dict[model] = str(Path(score_list[0][0]).parents[1])

# Load the images
model_to_images_dict = {}
for model, path in PATH_dict.items():
    model_to_images_dict[model] = {cls: os.listdir(os.path.join(path, cls)) for cls in count_dict.keys()}

# Shuffle the images
for model in PATH_dict.keys():
    for cls in count_dict.keys():
        random.shuffle(model_to_images_dict[model][cls])

ensembled_images = {cls: [] for cls in count_dict.keys()}
model_class_selected_counter = {model: {cls: 0 for cls in count_dict.keys()} for model in PATH_dict.keys()}

for cls in tqdm(list(count_dict.keys())):
    # First convert the mapping
    # Original(RMD_scores): {(model, cls): [(image_path, RMD_score), ...]}
    # New: {model: RMD_score, ...}

    # For top-k
    image_rmd_scores = []
    for (model, cls_), images in RMD_scores.items():
        if cls_ == cls:
            for image, score in images:
                image_rmd_scores.append((model, image, score))

    # Get sample_path -> (model, score) mapping
    sample_model_RMD_mapping = {}
    for sample in image_rmd_scores:
        sample_model_RMD_mapping[sample[1]] = sample[0], sample[2] # model, score
    
    if TopK:
        sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=True)
        chosen_samples = [sample[0] for sample in sorted_data[:count_dict[cls]]]
    elif BottomK:
        sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=False)
        chosen_samples = [sample[0] for sample in sorted_data[:count_dict[cls]]]
    else:
        # Normalize and clip RMD scores
        scores = np.array([score[1] for score in sample_model_RMD_mapping.values()])
        # mean = np.mean(scores); std = np.std(scores)
        
        if clip:
            lower_clip = np.percentile(scores, lower_percentile)
            upper_clip = np.percentile(scores, upper_percentile)
            print(f"Lower clip: {lower_clip}, Upper clip: {upper_clip}")
            clipped_scores = np.clip(scores, lower_clip, upper_clip)
            if normalize:
                mean = np.mean(clipped_scores); std = np.std(clipped_scores)
                result_scores = (clipped_scores - mean) / std
            else:
                result_scores = clipped_scores
        else:
            result_scores = scores
        
        probabilities = softmax_with_temperature(result_scores, TEMPERATURE)
        if INVERSE:
            # To get the inverse probabilities, first handle the numerical instability
            if np.min(probabilities) < 0:
                probabilities -= np.min(probabilities)
            # Handle devision by zero
            if np.sum(probabilities) == 0:
                raise ValueError("All probabilities are zero")
            probabilities = 1 / probabilities
            probabilities /= np.sum(probabilities) # Normalize the probabilities

        chosen_samples = np.random.choice(list(sample_model_RMD_mapping.keys()), size=count_dict[cls], replace=False, p=probabilities)
    
    # Update the result dictionary and counter
    for sample_path in chosen_samples:
        ensembled_images[cls].append({'model': sample_model_RMD_mapping[sample_path][0], 'image': sample_path})
        model_class_selected_counter[sample_model_RMD_mapping[sample_path][0]][cls] += 1


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