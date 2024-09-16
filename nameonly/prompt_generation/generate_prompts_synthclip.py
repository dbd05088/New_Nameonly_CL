import re
import json
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

# DomainNet
prompt_json_path = './prompts/synthclip_PACS_100.json'
dataset_count = PACS_count
NUM_PROMPTS = 100

def generate_img_caption(client, cls):
    cls_tmp = cls.replace('_',' ')
    base_message = f"Your task is to write me an image caption that includes and visually describes a scene around a concept. Your concept is {cls_tmp}. Output one single grammatically correct caption that is no longer than 15 words. Do not output any notes, word counts, facts, etc. Output one single sentence only."
    
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

totalprompt_dict = {}
for cls in tqdm(dataset_count):
    totalprompt_dict[cls] = {'metaprompts': []}
    class_prompt_list = []
    
    for i in range(NUM_PROMPTS):
        img_caption = generate_img_caption(client,cls)
        
        totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': i,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': img_caption
                    }
                ]
            }
        )

with open(prompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
