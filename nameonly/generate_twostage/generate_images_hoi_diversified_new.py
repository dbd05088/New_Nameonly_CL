import os
import json
import argparse
import shutil
from get_image_onestage import model_selector, adjust_list_length
from tqdm import tqdm

debug = False
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-r', '--root_dir', type=str)
parser.add_argument('-s', '--start_class', type=int)
parser.add_argument('-e', '--end_class', type=int)
parser.add_argument('-p', '--prompt_dir')
args = parser.parse_args()

if debug:
    model = None
else:
    model = model_selector(args.model_name)

# Load prompt json
with open(args.prompt_dir, 'r') as f:
    dataset_list = json.load(f)

print(f"ID indices to generate: {args.start_class} ~ {args.end_class}")
list_to_generate = dataset_list[args.start_class:args.end_class+1]
for dataset_dict in tqdm(list_to_generate):
    id = dataset_dict['id']
    positive_prompts = dataset_dict['positive_prompts']
    negative_prompts = dataset_dict['negative_prompts']
    
    # Generate positive images
    pos_save_dir = os.path.join(args.root_dir, str(id), 'pos')
    if os.path.exists(pos_save_dir):
        print(f"Removing existing directory: {pos_save_dir}")
        shutil.rmtree(pos_save_dir)
    os.makedirs(pos_save_dir, exist_ok=True)
    for i, positive_prompt in enumerate(positive_prompts):
        print(f"Generating positive image for {id}, prompt: {positive_prompt}")
        while True:
            try:
                image = model.generate_one_image(positive_prompt)
                break
            except:
                print(f"Error occurred. Retrying...")
                continue
        image_path = os.path.join(pos_save_dir, str(i).zfill(6) + '.png')
        image.save(image_path, "JPEG")
    
    # Generate negative images
    neg_save_dir = os.path.join(save_dir, str(id), 'neg')
    if os.path.exists(neg_save_dir):
        print(f"Removing existing directory: {neg_save_dir}")
        shutil.rmtree(neg_save_dir)
    os.makedirs(neg_save_dir, exist_ok=True)
    for i, negative_prompt in enumerate(negative_prompts):
        print(f"Generating negative image for {id}, prompt: {negative_prompt}")
        while True:
            try:
                image = model.generate_one_image(negative_prompt)
                break
            except:
                print(f"Error occurred. Retrying...")
                continue
        image_path = os.path.join(neg_save_dir, str(i).zfill(6) + '.png')
        image.save(image_path, "JPEG")
    