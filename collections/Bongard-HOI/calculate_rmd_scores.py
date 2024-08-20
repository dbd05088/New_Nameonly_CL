import torch
import yaml
import os
import json
import natsort
import random
import numpy as np
import pickle
import argparse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

def calculate_features(image_paths, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(image_path)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 512).to(device))
            continue
        with torch.no_grad():
            image_input = processor(images=image, return_tensors="pt").to(device)
            image_feature = model.get_image_features(**image_input) # [1, 512]
            image_features.append(image_feature)

    # Normalize the features
    image_features = torch.cat(image_features, dim=0) # [N, 512]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # dim=-1 -> torch.Size([N, 1]) -> Normalize each data

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def compute_class_agnostic_params(image_features):
    # mu_agnostic: (512,) / cov_agnostic: (512, 512)
    mu_agnostic = image_features.mean(axis=0)
    cov_agnostic = (image_features - mu_agnostic).T @ (image_features - mu_agnostic)

    cov_agnostic = cov_agnostic / len(image_features)

    return mu_agnostic, cov_agnostic


def compute_class_specific_params(image_features, object_labels, action_labels):
    # mu_specific: (C, 512) / cov_specific: (C, 512, 512)
    # Image features, class_labels are already aligned
    pairs = list(zip(object_labels, action_labels))
    unique_pairs = list(set(pairs))
    mu_specific = {}
    cov_specific = []
    
    for object, action in unique_pairs:
        cls_features = image_features[(object_labels == object) & (action_labels == action)]
        mu_specific[(object, action)] = cls_features.mean(axis=0)
        cov_specific.append((cls_features - mu_specific[(object, action)]).T @ (cls_features - mu_specific[(object, action)]))
    

    # cov_specific should be averaged over all classes
    cov_specific = np.array(cov_specific).sum(axis=0) / len(image_features) # (512, 512)
    return mu_specific, cov_specific

def mahalanobis_distance_specific(image_features, mu_specific_dict, sigma_specific, class_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    mu_specific_list = [mu_specific_dict[cls] for cls in class_labels]
    mu_specific_matrix = np.array(mu_specific_list)

    # Calculate the mahalanobis distance
    diff = image_features - mu_specific_matrix
    distance = -1.0 * diff @ sigma_specific_inv @ diff.T

    return distance

def mahalanobis_distance_agnostic(image_features, mu_agnostic, sigma_agnostic):
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    diff = image_features - mu_agnostic
    distance = -1.0 * diff @ sigma_agnostic_inv @ diff.T

    return distance

def mahalanobis_distance_manually(image_features, mu_agnostic, sigma_agnostic, mu_specific_dict, sigma_specific, object_labels, action_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    
    # Calculate the mahalanobis distance
    RMD = np.zeros((image_features.shape[0]))
    for i in range(image_features.shape[0]):
        feature = image_features[i]; label = (object_labels[i], action_labels[i])
        difference_specific = feature - mu_specific_dict[label] # (512,)
        M_specific = -1.0 * difference_specific @ sigma_specific_inv @ difference_specific.T

        difference_agnostic = feature - mu_agnostic
        M_agnostic = -1.0 * difference_agnostic @ sigma_agnostic_inv @ difference_agnostic.T

        RMD[i] = M_specific - M_agnostic

    return RMD

def softmax_with_temperature(z, T): 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

if __name__ == "__main__":
    # object1/action1, object1/action2, ..., object2/action1, object2/action2, ...
    image_paths = {
        'sdxl': './images/hoi_sdxl',
        'floyd': './images/hoi_floyd',
        'cogview2': './images/hoi_cogview2',
        'sdturbo': './images/hoi_sdturbo',
    }
    json_save_path = './RMD_scores/hoi_generated.json'
    first_model = next(iter(image_paths)); first_path = image_paths[first_model]
    objects = os.listdir(first_path)
    
    # Generate an empty dictionary - model- object - action - image_paths
    images_dict = {model: {object: {} for object in objects} for model in image_paths.keys()}
    for model, object_action_dict in images_dict.items():
        for object, action_images_dict in object_action_dict.items():
            object_path = os.path.join(image_paths[model], object)
            actions = os.listdir(object_path)
            for action in actions:
                action_path = os.path.join(object_path, action)
                images = natsort.natsorted(os.listdir(action_path))
                images = [os.path.join(action_path, image) for image in images]
                action_images_dict[action] = images


    # Concat all images into one list and generate corresponding class lables
    concatenated_images = []
    object_labels = []; action_labels = []; model_labels = []
    for model, object_action_dict in images_dict.items():
        for object, action_images_dict in object_action_dict.items():
            for action, images in action_images_dict.items():
                concatenated_images += images
                object_labels += [object] * len(images)
                action_labels += [action] * len(images)
                model_labels += [model] * len(images)
    
    
    # Convert labels to numpy array
    object_labels = np.array(object_labels); action_labels = np.array(action_labels); model_labels = np.array(model_labels)
    # Calculate the features
    features = calculate_features(concatenated_images, model)
    mu_agnostic, cov_agnostic = compute_class_agnostic_params(features)
    mu_specific_dict, cov_specific = compute_class_specific_params(features, object_labels, action_labels)
    
    # RMD = distance_specific - distance_agnostic
    RMD = mahalanobis_distance_manually(features, mu_agnostic, cov_agnostic, mu_specific_dict, cov_specific, object_labels, action_labels)

    # Split RMD scores using model labels
    RMD_each_model = {}
    for model in image_paths.keys():
        RMD_each_model[model] = RMD[model_labels == model]
    
    # Generate model - object - action - image_path - RMD dictionary
    model_object_action_image_RMD = {}
    for model in image_paths.keys():
        model_object_action_image_RMD[model] = {}
        for object in objects:
            model_object_action_image_RMD[model][object] = {}
            for action in os.listdir(os.path.join(image_paths[model], object)):
                model_object_action_image_RMD[model][object][action] = {}
                indices = np.where((model_labels == model) & (object_labels == object) & (action_labels == action))[0]
                RMDs = RMD[indices]
                paths = np.array(concatenated_images)[indices]
                model_object_action_image_RMD[model][object][action] = [{"image_path": path, "score": score} for path, score in zip(paths, RMDs)]
    
    # Save the RMD scores as json
    with open(json_save_path, 'w') as f:
        json.dump(model_object_action_image_RMD, f)