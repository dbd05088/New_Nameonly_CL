import os, random, shutil
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

NUM_POS_ACTIONS = 7

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

prompt_json_path = '../../nameonly/prompt_generation/prompts/generated_LLM_sdxl_ver2.json'
base_path = './'
json_path = './RMD_scores/generated_LLM_sdxl_ver2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.json'
target_path = './images/generated_LLM_ver2_RMD'

prompt_json = json.load(open(prompt_json_path, 'r'))
with open(json_path, 'r') as f:
    RMD_scores = json.load(f)
    RMD_scores_original = deepcopy(RMD_scores)

# Parse RMD json file to get PATH dict
models = list(RMD_scores.keys())
PATH_dict = {}
for model in models:
    first_uid = next(iter(RMD_scores[model]))
    first_pos = next(iter(RMD_scores[model][first_uid]))
    relative_path = RMD_scores[model][first_uid][first_pos][0]['image_path'] # Get the first item
    PATH_dict[model] = os.path.join(base_path, str(Path(relative_path).parents[2]))

# Get object list
first_model = next(iter(PATH_dict)); first_path = PATH_dict[first_model]
# uids = os.listdir(first_path)
uids = ['225', '1095']

# Shuffle the images
for model in PATH_dict.keys():
    for uid in uids:
        random.shuffle(RMD_scores[model][uid]['pos'])
        random.shuffle(RMD_scores[model][uid]['neg'])

# First, convert prompt dictionary mapping to key - value pairs
    id_prompt_dict = {}
    for prompt_dict in prompt_json:
        id_prompt_dict[prompt_dict['id']] = {
            'object_class': prompt_dict['object_class'],
            'action_class': prompt_dict['action_class'],
            'positive_prompts': prompt_dict['positive_prompts'],
            'negative_prompts': prompt_dict['negative_prompts']
        }
# Get mapping (object, action) -> {'model1': [image1, image2, ...], 'model2': [image1, image2, ...], ...}
object_action_to_model_images = {}
for model, uid_images_dict in RMD_scores.items():
    for uid, pos_neg_dict in uid_images_dict.items():
        object_class = id_prompt_dict[int(uid)]['object_class'][0] # Only one object class
        action_classes = id_prompt_dict[int(uid)]['action_class']
        pos_image_path_scores = pos_neg_dict['pos']
        neg_image_path_scores = pos_neg_dict['neg']
        for i, image_path_score in enumerate(pos_image_path_scores):
            image_path = image_path_score['image_path']; score = image_path_score['score']
            action_class = action_classes[i]
            if (object_class, action_class) not in object_action_to_model_images:
                object_action_to_model_images[(object_class, action_class)] = []
            object_action_to_model_images[(object_class, action_class)].append({'image_path': image_path, 'score': score, 'model': model})
        for i, image_path_score in enumerate(neg_image_path_scores):
            image_path = image_path_score['image_path']; score = image_path_score['score']
            action_class = action_classes[i + NUM_POS_ACTIONS]
            if (object_class, action_class) not in object_action_to_model_images:
                object_action_to_model_images[(object_class, action_class)] = []
            object_action_to_model_images[(object_class, action_class)].append({'image_path': image_path, 'score': score, 'model': model})

model_image_count = {model: 0 for model in PATH_dict.keys()}
            
# Generate ensembled images
for prompt_dict in tqdm(prompt_json):
    id = prompt_dict['id']
    # Create a new directory for each prompt
    pos_path = os.path.join(target_path, str(id), 'pos'); neg_path = os.path.join(target_path, str(id), 'neg')
    os.makedirs(pos_path, exist_ok=True); os.makedirs(neg_path, exist_ok=True)

    object_class = prompt_dict['object_class'][0]
    action_classes = prompt_dict['action_class']
    image_idx = 0
    for pos_action in action_classes[:NUM_POS_ACTIONS]:
        sample_RMD_scores = object_action_to_model_images[(object_class, pos_action)]
        scores = [sample['score'] for sample in sample_RMD_scores]
        if clip:
            lower_clip = np.percentile(scores, lower_percentile)
            upper_clip = np.percentile(scores, upper_percentile)
            clipped_scores = np.clip(scores, lower_clip, upper_clip)
            if normalize:
                mean = np.mean(clipped_scores); std = np.std(clipped_scores)
                result_scores = (clipped_scores - mean) / std
            else:
                result_scores = clipped_scores
        else:
            result_scores = scores
        probabilities = softmax_with_temperature(result_scores, TEMPERATURE)
        chosen_samples = np.random.choice(sample_RMD_scores, size=1, replace=False, p=probabilities)
        sample_RMD_scores.remove(chosen_samples[0])
        model = chosen_samples[0]['model']; image_path = chosen_samples[0]['image_path']
        model_image_count[model] += 1
        
        # Copy the image to the target directory
        image_name = f"{str(image_idx).zfill(6)}_{model}.png"
        image_idx += 1
        shutil.copy(image_path, os.path.join(pos_path, image_name))
    
    image_idx = 0
    for neg_action in action_classes[NUM_POS_ACTIONS:]:
        sample_RMD_scores = object_action_to_model_images[(object_class, neg_action)]
        scores = [sample['score'] for sample in sample_RMD_scores]
        if clip:
            lower_clip = np.percentile(scores, lower_percentile)
            upper_clip = np.percentile(scores, upper_percentile)
            clipped_scores = np.clip(scores, lower_clip, upper_clip)
            if normalize:
                mean = np.mean(clipped_scores); std = np.std(clipped_scores)
                result_scores = (clipped_scores - mean) / std
            else:
                result_scores = clipped_scores
        else:
            result_scores = scores
        probabilities = softmax_with_temperature(result_scores, TEMPERATURE)
        chosen_samples = np.random.choice(sample_RMD_scores, size=1, replace=False, p=probabilities)
        sample_RMD_scores.remove(chosen_samples[0])
        model = chosen_samples[0]['model']; image_path = chosen_samples[0]['image_path']
        model_image_count[model] += 1
        
        # Copy the image to the target directory
        image_name = f"{str(image_idx).zfill(6)}_{model}.png"
        image_idx += 1
        shutil.copy(image_path, os.path.join(neg_path, image_name))
        
print(model_image_count)