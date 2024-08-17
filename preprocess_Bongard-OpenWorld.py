# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import itertools
import jsonlines
import numpy as np

np.random.seed(42)

def save_dataset(dataset_name, input_folder, output_folder, subset_name, seed=None):
    subset_folder = os.path.join(input_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    
    # if subset_name == "test":
    #     postfix = f"{dataset_name}_{subset_name}"
    # else:
    #     postfix = f"{dataset_name}_{subset_name}_seed{seed}"
    with open(f"{input_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    json_data_list = []
    
    for item in datalist:
        # 6 positive imgs, 1 positive query, 6 negative imgs, 1 negative query
        # --> 2 pos, 2 neg
        # how many data from this single item?
        
        answer = item['caption']
        positive_imgfiles = item['imageFiles'][:7]
        negative_imgfiles = item['imageFiles'][7:]
        positive_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in positive_imgfiles]
        negative_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in negative_imgfiles]
        
        positive_files = positive_imgfiles[:-1]
        positive_queryfile = positive_imgfiles[-1]
        
        negative_files = negative_imgfiles[:-1]
        negative_queryfile =  negative_imgfiles[-1]
        
        arr = np.arange(len(positive_files))
        nCr = list(itertools.combinations(arr, num_per_set))
        for idx, index in enumerate(nCr):
            index = np.array(list(index))
            imgs = [positive_files[i] for i in index] + [negative_files[i] for i in index]
        
            # Structure for LLaVA JSON
            json_data = {
                "id": item['uid'] + "-" + str(idx),
                "image": imgs + [positive_queryfile],#" |sep| ".join(imgs),
                "commonSense":item["commonSense"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\nQuery: <image>\n" + prompts + '\nChoice list:[Positive, Negative]. Your answer is:'
                        # "value": "<image>"*len(imgs) + "\n" + prompts
                    },
                    { 
                        "from": "gpt",
                        "value": "Positive"
                    }
                ]
            }
            json_data_list.append(json_data)
            
            imgs = np.random.choice(positive_files, size=num_per_set, replace=False).tolist() + np.random.choice(negative_files, size=num_per_set, replace=False).tolist()
            
            # breakpoint()
            json_data = {
                "id": item['uid'] + "-" + str(idx),
                "commonSense":item["commonSense"],
                "image": imgs + [negative_queryfile],#" |sep| ".join(imgs),
                "conversations": [
                    {
                        "from": "human",
                        "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\nQuery: <image>\n" + prompts + '\nChoice list:[Positive, Negative]. Your answer is:'
                        # "value": "<image>"*len(imgs) + "\n" + prompts
                    },
                    { 
                        "from": "gpt",
                        "value": "Negative"
                    }
                ]
            }
            json_data_list.append(json_data)
                
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name + ".json")
    print(len(json_data_list))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example

# preprocess jsonl to json
# train (combine train and val)
# train_data = []
# with jsonlines.open(f"{output_folder}/train.jsonl") as f:
#     for line in f.iter():
#         train_data.append(line)

# with jsonlines.open(f"{output_folder}/val.jsonl") as f:
#     for line in f.iter():
#         train_data.append(line)

# with open(f"{output_folder}/train.json", 'w') as json_file:
#     json.dump(train_data, json_file, indent=4)  

# test
# test_data = []
# with jsonlines.open(f"{output_folder}/test.jsonl") as f:
#     for line in f.iter():
#         test_data.append(line)

# with open(f"{output_folder}/test.json", 'w') as json_file:
#     json.dump(test_data, json_file, indent=4)

# User inputs
num_per_sets = [2,3,4]
seeds = [1,2,3,4,5]
types = "ma"
input_folder = f'collections/Bongard-OpenWorld/{types}'
output_folder = 'collections/Bongard-OpenWorld'
train_seed = 1

for num_per_set in num_per_sets:
    for seed in seeds:
        input_folder = f'collections/Bongard-OpenWorld/{types}_splits'
        output_folder = f'collections/Bongard-OpenWorld/{types}_more/{num_per_set*2+1}_set'
        os.makedirs(output_folder, exist_ok=True)
        prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where "positive" images share "common" visual concepts and "negative" images cannot, the "common" visual concepts exclusively depicted by the "positive" images. And then given 1 "query" image, please determine whether it belongs to "positive" or "negative".'''

        ### Only for MA ###
        if types == "ma":
            save_dataset('Bongard-OpenWorld', input_folder, output_folder, 'Bongard-OpenWorld_test')
        save_dataset('Bongard-OpenWorld', input_folder, output_folder, f'Bongard-OpenWorld_train_seed{seed}')
