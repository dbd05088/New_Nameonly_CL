# 민혁님께서 json 달라고 하신 것 (txt file을 json으로 바꿈)
import json
from classes import *

txt_path = './prompts/cafo_cct.txt'
json_path = './prompts/cafo_cct.json'

num_classes = 12
num_prompts_per_cls = 50
with open(txt_path, 'r') as f:
    text_file = f.readlines()
text_file = [txt.strip() for txt in text_file]

json_dict = {}
for i in range(num_classes):
    cls_index = i * (num_prompts_per_cls + 1)
    prompt_start_idx = cls_index + 1
    prompt_end_idx = prompt_start_idx + num_prompts_per_cls - 1
    cls_name = text_file[cls_index]
    print(f"Class: {cls_name}, prompt_start: {prompt_start_idx}, prompt_end: {prompt_end_idx}")
    prompts_list = text_file[prompt_start_idx:prompt_end_idx + 1]
    json_dict[cls_name] = prompts_list

with open(json_path, 'w') as f:
    json.dump(json_dict, f)
    