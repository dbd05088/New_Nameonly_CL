import os
import json
import argparse
from get_image_onestage import model_selector
from tqdm import tqdm
from utils import sanitize_filename

debug = False
model_name = "sdxl"

if debug:
    model = None
else:
    model = model_selector(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root_dir', type=str)
parser.add_argument('-s', '--start_class', type=int)
parser.add_argument('-e', '--end_class', type=int)
parser.add_argument('-n', '--num_images', type=int, default=14)
args = parser.parse_args()

num_images = args.num_images

# Process jsonl file
data_list = []
with open('./train.jsonl', 'r') as f:
    for line in f:
        json_object = json.loads(line)
        data_list.append(json_object)

print(f"Class indices to generate: {args.start_class} ~ {args.end_class}")
list_to_generate = data_list[args.start_class:args.end_class+1]
for line in tqdm(list_to_generate):
    uid = line['uid']; commonSense = line['commonSense']; concept = line['concept']; caption = line['caption']
    save_dir = os.path.join(args.root_dir, uid)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(num_images):
        print(f"Generating image for uid:{uid}, caption:{caption}")
        image = model.generate_one_image(caption)
        image_name = f"{sanitize_filename(concept)}_{str(i).zfill(6)}"
        image_path = os.path.join(save_dir, image_name + '.png')
        image.save(image_path, "JPEG")
