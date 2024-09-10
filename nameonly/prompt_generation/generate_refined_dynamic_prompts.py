import re
import json
import nltk
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

# important
raw_prompt_json_path = './prompts/gpt4_hierarchy_cot_4_PACS.json' # this should be dynamic
refined_prompt_json_path = './prompts/gpt4_hierarchy_cot_4_PACS_refined.json'
dataset_count = PACS_count
# important

def generate_refined_caption(client, cls, caption):
    cls_tmp = cls.replace('_',' ')
    base_message = f"To generate images using a text-to-image generation model, I've generated prompts. However, some sentences aren't appropriate for the concept, so I want to refine them into sentences that contain context suitable and reasonable for this concept. Both appropriate and inappropriate captions will be provided as input. Your caption is {caption}, and your concept is {cls_tmp} Output one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."
    
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

with open(raw_prompt_json_path, 'r') as file:
    data = json.load(file)

totalprompt_dict = {}
for cls in tqdm(dataset_count):
    totalprompt_dict[cls] = {'metaprompts': []}
    class_prompt_list = []
    
    idx = 0
    for i in range(10):
        for j in range(5):            
            caption_old = data[cls]['metaprompts'][i]['prompts'][j]['content']
            caption_new = generate_refined_caption(client, cls, caption_old)
            
            totalprompt_dict[cls]['metaprompts'].append(
                {
                'index': idx,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': caption_new
                    }
                ]
            }
            )
            idx+=1

with open(refined_prompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
