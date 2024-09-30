import os
import json
import random
import re
import ast
from tqdm import tqdm
from pathlib import Path

def extract_number(filename):
    match = re.search(r'/(\d+)_', filename)
    return int(match.group(1)) if match else 0

def find_dict_by_id(data, target_id):
    for item in data:
        if 'id' in item and item['id'] == target_id:
            return item
    return None

seed = 5
base_json_path = f'./ma_splits/Bongard-HOI_train_seed{str(seed)}.json'
base_prompt_path = '../../nameonly/prompt_generation/prompts/generated_LE_ver1.json'
generated_image_path = './images/generated_LE_ver1_RMD'
output_path = f"./generated_LE_ver1_RMD_splits"
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(base_json_path, 'r') as f:
    train_datalist = json.load(f)
with open(base_prompt_path, 'r') as f:
    prompt_dict = json.load(f)

result_list = []
for base_data in tqdm(train_datalist):
    result_dict = {}
    result_dict['id'] = base_data['id']
    result_dict['type'] = base_data['type']
    result_dict['image_files'] = []
    result_dict['object_class'] = base_data['object_class']
    image_path = os.path.join(generated_image_path, str(base_data['id']))
    pos_image_list = [os.path.join(image_path, 'pos', image) for image in os.listdir(os.path.join(image_path, 'pos'))]
    pos_image_list = sorted(pos_image_list, key=extract_number)
    neg_image_list = [os.path.join(image_path, 'neg', image) for image in os.listdir(os.path.join(image_path, 'neg'))]
    neg_image_list = sorted(neg_image_list, key=extract_number)
    result_dict['image_files'].extend(pos_image_list + neg_image_list)
    
    # Find action class from prompt_dict
    prompt_data = find_dict_by_id(prompt_dict, base_data['id'])
    result_dict['action_class'] = prompt_data['action_class']
    result_dict['action_object'] = f"{prompt_data['action_class'][0]}_{base_data['object_class'][0]}"
    result_dict['action'] = result_dict['action_class'][0]

    result_list.append(result_dict)
    
output_json_path = os.path.join(output_path, f'Bongard-HOI_train_seed{seed}.json')
with open(output_json_path, 'w') as f:
    json.dump(result_list, f)
print(f"Saved to {output_json_path}")