import os, random, shutil
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

# NORMALIZATION, CLIPPING
normalize = True # Fix
clip = True # Fix
lower_percentile = 5.0 # 5.0
# lower_percentile = 2.5 # 5.0
upper_percentile = 95.0 # 95.0
# upper_percentile = 97.5 # 95.0

equalweight = False
TopK = False
BottomK = False
INVERSE = False
TEMPERATURE = 0.5

base_path = './'
json_path = './RMD_scores/hoi_generated.json'
target_path = './images/generated_RMD'

with open(json_path, 'r') as f:
    RMD_scores = json.load(f)
    RMD_scores_original = deepcopy(RMD_scores)

# Parse RMD json file to get PATH dict
models = list(RMD_scores.keys())
PATH_dict = {}
for model in models:
    first_object = next(iter(RMD_scores[model]))
    first_action = next(iter(RMD_scores[model][first_object]))
    relative_path = RMD_scores[model][first_object][first_action][0]['image_path'] # Get the first item
    PATH_dict[model] = os.path.join(base_path, str(Path(relative_path).parents[2]))

# Get object list
first_model = next(iter(PATH_dict)); first_path = PATH_dict[first_model]
objects = os.listdir(first_path)

# Shuffle the images
for model in PATH_dict.keys():
    for object in objects:
        for action in RMD_scores[model][object]:
            random.shuffle(RMD_scores[model][object][action])

ensembled_images = {object: {action: [] for action in RMD_scores[first_model][object]} for object in objects}
model_object_action_selected_counter = {model: {object: {action: 0 for action in RMD_scores[first_model][object]} for object in objects} for model in PATH_dict.keys()}

for object in objects:
    for action in RMD_scores[first_model][object]:
        NUM_IMAGES = len(RMD_scores_original[models[0]][object][action])
        # For top-k, get list of [(model, image_path, score), ...]
        image_rmd_scores = []
        for model, object_action_dict in RMD_scores.items():
            image_rmd_scores += [(model, os.path.join(base_path, image['image_path']), image['score']) for image in object_action_dict[object][action]]
        
        # Get sample_path -> (model, score) mapping
        sample_model_RMD_mapping = {}
        for sample in image_rmd_scores:
            sample_model_RMD_mapping[sample[1]] = sample[0], sample[2] # model, score
        
        if TopK:
            sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=True)
            chosen_samples = [sample[0] for sample in sorted_data[:NUM_IMAGES]]
        elif BottomK:
            sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=False)
            chosen_samples = [sample[0] for sample in sorted_data[:NUM_IMAGES]]
        elif equalweight:
            probabilities = [1 / len(PATH_dict)] * len(PATH_dict)
            chosen_samples = []
            while True:
                chosen_model = random.choices(models, weights=probabilities, k=1)[0]
                if len(RMD_scores[chosen_model][object][action]) > 0:
                    chosen_image = RMD_scores[chosen_model][object][action].pop()
                    chosen_image_path = os.path.join(base_path, chosen_image['image_path'])
                    chosen_samples.append(chosen_image_path)
                if len(chosen_samples) == NUM_IMAGES:
                    print(f"Break for object {object} and action {action} with {len(chosen_samples)} images")
                    break
        else:
            # Normalize and clip RMD scores
            scores = np.array([score[1] for score in sample_model_RMD_mapping.values()])
            # mean = np.mean(scores); std = np.std(scores)
            
            if clip:
                lower_clip = np.percentile(scores, lower_percentile)
                upper_clip = np.percentile(scores, upper_percentile)
                # print(f"Lower clip: {lower_clip}, Upper clip: {upper_clip}")
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
                probabilities /= np.sum(probabilities)
            
            chosen_samples = np.random.choice(list(sample_model_RMD_mapping.keys()), size=NUM_IMAGES, replace=False, p=probabilities)
                
        # Update the result dictionary and counter
        for sample_path in chosen_samples:
            ensembled_images[object][action].append({'model': sample_model_RMD_mapping[sample_path][0], 'image': sample_path})
            model_object_action_selected_counter[sample_model_RMD_mapping[sample_path][0]][object][action] += 1
        
# Check the number of images selected for each model, for each object, for each action
for model, object_action_counter in model_object_action_selected_counter.items():
    model_count = sum([sum(action_counter.values()) for action_counter in object_action_counter.values()])
    print(f"Model {model} has {model_count} images selected")

# Sanity check the number of images for each object, for each action
for object, action_dict in ensembled_images.items():
    for action, images in action_dict.items():
        if len(images) != len(RMD_scores_original[models[0]][object][action]):
            raise ValueError(f"uid {object} action {action} has {len(RMD_scores_original[models[0]][object][action])} images but {len(images)} images are selected")

for object, action_dict in ensembled_images.items():
    for action, images in action_dict.items():
        image_counter = 0
        target_object_action_path = os.path.join(target_path, object, action)
        os.makedirs(target_object_action_path, exist_ok=True)
        for img in images:
            model = img['model']; image_name = img['image']; prompt = '_'.join(Path(image_name).stem.split('_')[:-1])
            new_image_name = f"{model}_{prompt}_{str(image_counter).zfill(6)}.png"
            image_counter += 1
            shutil.copy(image_name, os.path.join(target_object_action_path, new_image_name))