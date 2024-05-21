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
# count_dict = PACS_count
# rmd_pickle_path = './RMD_scores/PACS_final_generated_RMD.pkl'
# rmd_pickle_path = './RMD_scores/cct_generated_RMD.pkl'
# rmd_pickle_path = './RMD_scores/DomainNet_generated_RMD.pkl'


# target_path = './datasets/neurips/PACS/final/PACS_final_web_all_samples_prob_temp_0_5'
# target_path = '../dataset/PACS_final/PACS_final_generated_RMD_w_normalize_clip_90_temp_0_25'
# target_path = '../dataset/cct/cct_generated_RMD_w_normalize_clip_90_temp_0_25'
# target_path = './datasets/neurips/new_generated/DomainNet/DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5'

# PACS web from large
count_dict = PACS_count
rmd_pickle_path = './RMD_scores/PACS_final_web_from_large.pkl'
target_path = './datasets/neurips/web/PACS/PACS_final_web_from_large_RMD_w_normalize_clip_90_temp_0_25'
PATH_dict = {
    'flickr': './datasets/neurips/web/PACS/PACS_flickr_from_large_filtered',
    'google': './datasets/neurips/web/PACS/PACS_google_from_large_filtered',
    'bing': './datasets/neurips/web/PACS/PACS_bing_from_large_filtered'
}

# PACS_final generated
# PATH_dict = {
#         'sdxl': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_sdxl_subsampled_filtered',
#         'dalle2': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_dalle2_subsampled_filtered',
#         'floyd': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_floyd_subsampled_filtered',
#         'cogview2': './datasets/neurips/new_generated/PACS/PACS_final_static_cot_50_cogview2_subsampled',
# }

# # DomainNet 
# PATH_dict = {
#     'sdxl': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_sdxl_subsampled_filtered',
#     'dalle2': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_dalle2_subsampled_filtered',
#     'floyd': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_floyd_subsampled_filtered',
#     'cogview2': './datasets/neurips/new_generated/DomainNet/DomainNet_static_cot_50_cogview2_subsampled_filtered'
# }

# # cct generated
# PATH_dict = {
#     'sdxl': './datasets/neurips/new_generated/cct/cct_static_cot_50_sdxl_subsampled_filtered',
#     'dalle2': './datasets/neurips/new_generated/cct/cct_static_cot_50_dalle2_subsampled_filtered',
#     'floyd': './datasets/neurips/new_generated/cct/cct_static_cot_50_floyd_subsampled_filtered',
#     'cogview2': './datasets/neurips/new_generated/cct/cct_static_cot_50_cogview2_subsampled'
# }


# PATH_dict = {
#         'flickr': './datasets/neurips/PACS/PACS_flickr_subsampled_filtered',
#         'google': './datasets/neurips/PACS/PACS_google_subsampled_filtered',
#         'bing': './datasets/neurips/PACS/PACS_bing_subsampled_filtered',
# }



# PATH_dict = {
#     'sdxl': './datasets/neurips/PACS/PACS_sdxl_diversified_subsampled',
#     'dalle2': './datasets/neurips/PACS/PACS_dalle2_subsampled',
#     'floyd': './datasets/neurips/PACS/PACS_floyd_subsampled',
#     'cogview2': './datasets/neurips/PACS/PACS_cogview2_subsampled',
# }
# PATH_dict = {
#         'flickr': './datasets/neurips/cct/cct_flickr_subsampled_filtered',
#         'google': './datasets/neurips/cct/cct_google_subsampled_filtered',
#         'bing': './datasets/neurips/cct/cct_bing_subsampled_filtered',
# }

# PATH_dict = {
#     'flickr': './PACS_final_for_rmd_check'
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

    # Get sample_path -> score mapping
    sample_model_RMD_mapping = {}
    for sample in image_rmd_scores:
        sample_model_RMD_mapping[sample[1]] = sample[0], sample[2] # model, score
    
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
        
    # if normalize:
    #     normalized_scores = (scores - mean) / std
    #     lower_clip = np.percentile(normalized_scores, lower_percentile)
    #     upper_clip = np.percentile(normalized_scores, upper_percentile)
    #     if clip:
    #         normalized_scores = np.clip(normalized_scores, lower_clip, upper_clip)
    # else:
    #     normalized_scores = scores
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