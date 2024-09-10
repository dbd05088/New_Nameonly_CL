import re
import os
import json
import nltk
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy

random.seed(42)

# important
static_prompt_json_path = './prompts/gpt4_hierarchy_cot_4.json'
refined_prompt_json_path = './prompts/gpt4_hierarchy_cot_4_DomainNet_refined.json'
dataset_count = DomainNet_count

# Load static prompt
with open(static_prompt_json_path, 'r') as f:
    static_prompt = json.load(f)

# Convert static prompt to dynamic prompt
dynamic_prompt_dict = {}
for cls in dataset_count:
    dynamic_prompt_dict[cls] = {'metaprompts': deepcopy(static_prompt['metaprompts'])}

for k, v in dynamic_prompt_dict.items():
    metaprompt_list = v['metaprompts']
    for metaprompt_dict in metaprompt_list:
        if '[concept]' in metaprompt_dict['metaprompt']:
            metaprompt_dict['metaprompt'] = metaprompt_dict['metaprompt'].replace("[concept]", k)

        for prompt_dict in metaprompt_dict['prompts']:
            assert "[concept]" in prompt_dict['content'], f"No [concept] in {prompt_dict}"
            prompt_dict['content'] = prompt_dict['content'].replace("[concept]", k.replace('_', ' '))


def generate_refined_caption(client, concept, prompt):
    cls_tmp = cls.replace('_',' ')
    base_message = f"To generate an image using a text-to-generation model, I created a prompt based on the concept. However, some sentences have contexts that do not fit the concept, so I would like to improve them into sentences with appropriate contexts. For example, when given the sentence 'A photo of a whale in the football field,' this has a context that does not fit the concept of a football field. This can be improved to 'A photo of a whale in the ocean.' When given a concept and the corresponding prompt, please refine it into a sentence with an appropriate context.\nconcept:{concept}\nprompt:{prompt}\nOutput one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message}
        ]
    )
    response_content = response.choices[0].message.content
    
    return response_content # assume work well w/ condition: Output one single sentence only.


client = OpenAI(api_key="sk-proj-bPJxpKwauBBFBZJw7nEgT3BlbkFJePaQfARB48iyTbZfxSXg")

# Open refined prompt json file
if os.path.exists(refined_prompt_json_path):
    with open(refined_prompt_json_path, 'r') as f:
        totalprompt_dict = json.load(f)
else:
    totalprompt_dict = {}

for cls, cls_prompt_dict in tqdm(dynamic_prompt_dict.items()):
    if cls in totalprompt_dict:
        print(f"Skipping {cls}")
        continue
        
    cls_prompt_list = cls_prompt_dict['metaprompts']
    for metaprompt_dict in tqdm(cls_prompt_list):
        for prompt_dict in metaprompt_dict['prompts']:
            print(f"Before: {prompt_dict['content']}")
            new_prompt = generate_refined_caption(client, cls, prompt_dict['content'])
            prompt_dict['content'] = new_prompt
            print(f"After: {prompt_dict['content']}\n")
            
    totalprompt_dict[cls] = cls_prompt_dict
    with open(refined_prompt_json_path, 'w') as f:
        json.dump(totalprompt_dict, f)