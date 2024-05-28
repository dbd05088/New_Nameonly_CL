# 이제는 equalweight용
import os, random, shutil
import pickle
import numpy as np
from classes import *
from tqdm import tqdm

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

Equalweight = True
TopK = False
BottomK = False

INVERSE = False
TEMPERATURE = 2

# NICO
count_dict = cifar10_count
rmd_pickle_path = './RMD_scores/cifar10_generated.pkl'
target_path = '/home/user/seongwon/New_Nameonly_CL/nameonly/datasets/neurips/new_generated/cifar10_generated_equalweight'
PATH_dict = {
    'sdxl': './datasets/neurips/new_generated/cifar10/cifar10_static_cot_50_sdxl_realistic_subsampled_filtered',
    'dalle2': './datasets/neurips/new_generated/cifar10/cifar10_static_cot_50_dalle2_realistic_subsampled_filtered',
    'floyd': './datasets/neurips/new_generated/cifar10/cifar10_static_cot_50_floyd_realistic_subsampled_filtered',
    'cogview2': './datasets/neurips/new_generated/cifar10/cifar10_static_cot_50_cogview2_realistic_subsampled_filtered'
}

# # PACS web from large
# count_dict = PACS_count
# rmd_pickle_path = './RMD_scores/PACS_final_web_from_large2.pkl'
# target_path = './datasets/neurips/web/PACS/PACS_final_web_from_large2_equalweight'
# PATH_dict = {
#     'flickr': './datasets/neurips/web/PACS/PACS_flickr_from_large2_filtered',
#     'google': './datasets/neurips/web/PACS/PACS_google_from_large2_filtered',
#     'bing': './datasets/neurips/web/PACS/PACS_bing_from_large2_filtered'
# }

# # DomainNet 
# PATH_dict = {
#     'sdxl': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_sdxl_subsampled_filtered',
#     'dalle2': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_dalle2_subsampled_filtered',
#     'floyd': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_floyd_subsampled_filtered',
#     'cogview2': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_cogview2_subsampled_filtered'
# }

# PACS_final
# PATH_dict = {
#         'sdxl': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_sdxl_subsampled_filtered',
#         'dalle2': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_dalle2_subsampled_filtered',
#         'floyd': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_floyd_subsampled_filtered',
#         'cogview2': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_cogview2_subsampled',
# }

# cct
# PATH_dict = {
#     'sdxl': './datasets/neurips/new_generated/cct/cct_static_cot_50_sdxl_subsampled_filtered',
#     'dalle2': './datasets/neurips/new_generated/cct/cct_static_cot_50_dalle2_subsampled_filtered',
#     'floyd': './datasets/neurips/new_generated/cct/cct_static_cot_50_floyd_subsampled_filtered',
#     'cogview2': './datasets/neurips/new_generated/cct/cct_static_cot_50_cogview2_subsampled'
# }

# PACS before
# PATH_dict = {
#     'sdxl': './datasets/neurips/PACS/PACS_sdxl_diversified_subsampled',
#     'dalle2': './datasets/neurips/PACS/PACS_dalle2_subsampled',
#     'floyd': './datasets/neurips/PACS/PACS_floyd_subsampled',
#     'cogview2': './datasets/neurips/PACS/PACS_cogview2_subsampled',
# }

# Load the RMD scores
with open(rmd_pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

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
    
    if TopK:
        image_rmd_scores.sort(key=lambda x: x[2], reverse=True)
        for i in range(count_dict[cls]):
            model_name = image_rmd_scores[i][0]; path = image_rmd_scores[i][1]
            ensembled_images[cls].append({'model': model_name, 'image': path})
            model_class_selected_counter[model_name][cls] += 1

    elif BottomK:
        image_rmd_scores.sort(key=lambda x: x[2], reverse=False)
        for i in range(count_dict[cls]):
            model_name = image_rmd_scores[i][0]; path = image_rmd_scores[i][1]
            ensembled_images[cls].append({'model': model_name, 'image': path})
            model_class_selected_counter[model_name][cls] += 1
    
    else:
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