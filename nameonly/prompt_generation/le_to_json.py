import pickle
import json
from classes import *

dataset_count = DomainNet_count
pickle_path = './prompts/prompt_LE_DomainNet.pkl'
json_path = './prompts/LE_100_DomainNet.json'
LE_dict = pickle.load(open(pickle_path, 'rb'))

result_dict = {}
for cls in dataset_count.keys():
    result_dict[cls] = {
        "metaprompts": [
            {
                "index": 0,
                "metaprompt": "dummy",
                "prompts": []
            }
        ]
    }
    
    for i, prompt in enumerate(LE_dict[cls]):
        result_dict[cls]["metaprompts"][0]["prompts"].append(
            {
                "index": i,
                "content": prompt
            }
        )
    

with open(json_path, 'w') as f:
    json.dump(result_dict, f, indent=4)