import os
import json
import random
import re
import ast
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="sk-proj-bPJxpKwauBBFBZJw7nEgT3BlbkFJePaQfARB48iyTbZfxSXg")
json_path = './train.json'
output_path = './train_generated_RMD_new.json'
samples_path = './images/generated_RMD_new'
with open(json_path, 'r') as f:
    data_dict = json.load(f)
    
def select_action(action_set, positive_action):
    base_message = f"To train a model that distinguishes between positive and negative images, you need to choose 7 negative actions from the following negative action list. To help the model distinguish between positive and negative actions, it is important to choose hard negative actions.\npositive action: {positive_action}\nnegative action list: {action_set}\n\nPlease select a total of 7 negative actions, and if the length of the negative action list is less than 7, you can allow duplicates. The response format must be strictly result: [negative_action1, negative_action2, ...].\n"
    
    # Generate 7 negative actions
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "user", "content": base_message},
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"\[.*\]", response_content)
            result_list = ast.literal_eval(match.group())
        except Exception as e:
            print(e)
            continue
        
        if len(result_list) == 7 and all(action in action_set for action in result_list):
            break
    
    return result_list
    
result_list = []
for data in tqdm(data_dict):
    generated_dict = {}
    generated_dict['id'] = data['id']
    generated_dict['type'] = data['type']
    generated_dict['image_files'] = []
    generated_dict['object_class'] = data['object_class']
    generated_dict['action_class'] = []
    
    # Randomly sample images
    object_class = data['object_class'][0]
    
    # Assign positive samples to positive action
    positive_action = data['action_class'][0]
    generated_dict['action_class'].extend([positive_action] * 7)
    positive_sample_path = os.path.join(samples_path, object_class, positive_action)
    positive_samples_list = os.listdir(positive_sample_path)
    positive_samples_list = [os.path.join(positive_sample_path, sample) for sample in positive_samples_list]
    positive_samples = random.sample(positive_samples_list, 7)
    generated_dict['image_files'].extend(positive_samples)
    
    # Assign negative samples to negative action
    actions = os.listdir(os.path.join(samples_path, object_class))
    actions.remove(positive_action)
    
    # Select 7 negative actions using GPT-4o
    # negative_action_list = select_action(actions, positive_action)
    negative_action_list = random.choices(actions, k=7)
    
    selected_samples = set()
    for negative_action in negative_action_list:
        generated_dict['action_class'].append(negative_action)
        negative_sample_path = os.path.join(samples_path, object_class, negative_action)
        negative_samples_list = os.listdir(negative_sample_path)
        negative_samples_list = [os.path.join(negative_sample_path, sample) for sample in negative_samples_list]
        available_samples = [sample for sample in negative_samples_list if sample not in selected_samples]
        
        if available_samples:
            negative_samples = random.choice(available_samples)
            selected_samples.add(negative_samples)
            generated_dict['image_files'].append(negative_samples)
        else:
            # Select any negative sample
            negative_samples = random.choice(negative_samples_list)
            generated_dict['image_files'].append(negative_samples)
            
    # Assign the rest of the samples to the rest of the actions
    result_list.append(generated_dict)

with open(output_path, 'w') as f:
    json.dump(result_list, f)