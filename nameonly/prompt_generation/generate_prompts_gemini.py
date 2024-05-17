import re
import pickle
import json
import google.generativeai as genai
import time
from tqdm import tqdm
from classes import *

GOOGLE_API_KEY="AIzaSyBFJXbGVaguzppf7qnnQvOlXmTywSfQRtM"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def generate_prompt_stage1(previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain realistic. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap (style, color, etc.) with these for the sake of diversity."
    end_message = f"\nPlease create one prompt sentence (under 10 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept]', which denotes the class name.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    response = model.generate_content(base_message)
    response = response._result.candidates[0].content.parts[0].text
    match = re.search(r'prompt:\s*(.*)', response, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response


def generate_prompt_stage2(metaprompt, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain realistic. The prompt should be similar to '{metaprompt}' but slightly different. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap (style, color, etc.) with these for the sake of diversity."
    
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept], which denotes the class name.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    response = model.generate_content(base_message)
    response = response._result.candidates[0].content.parts[0].text
    match = re.search(r'prompt:\s*(.*)', response, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response

cls_count_dict = PACS_count
metaprompt_json_path = './prompts/gemini_temp.json'
totalprompt_json_path = './prompts/gemini_static_totalprompts_with_cot_50.json'

num_metaprompts = 10
num_prompts_per_metaprompt = 5



# # Metaprompt generation
# metaprompts = ['A photo of a [concept].', 'A photo of [concept] in neutral tones']
# for i in tqdm(range(num_metaprompts - 2)):
#     try:
#         prompt = generate_prompt_stage1(metaprompts)
#         print(f"Previous prompt list: {metaprompts}")
#         print(f"Generated metaprompt for stage: {prompt}")
#         metaprompts.append(prompt)
#     except:
#         pass

# with open(metaprompt_json_path, 'w') as f:
#     json.dump(metaprompts, f)


# Diversified prompt generation
# Load metaprompt pickle file
with open(metaprompt_json_path, 'r') as f:
    metaprompt_list = json.load(f)

totalprompt_dict = {'metaprompts': []}
for i, metaprompt in enumerate(tqdm(metaprompt_list)):
    metaprompt_dict = {'index': i, 'metaprompt': metaprompt, 'prompts': []}
    diversified_prompts = []
    for j in range(num_prompts_per_metaprompt):
        prompt = generate_prompt_stage2(metaprompt, diversified_prompts)
        print(f"previous generated prompts: {diversified_prompts}")
        print(f"Generated prompt: {prompt}")
        diversified_prompts.append(prompt)
        metaprompt_dict['prompts'].append({'index': j, 'content': prompt})
        time.sleep(5)
    totalprompt_dict['metaprompts'].append(metaprompt_dict)

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
