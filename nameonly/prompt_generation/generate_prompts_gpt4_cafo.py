import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *

def generate_prompt_cafo(client, concept_name):
    message = f"""I need to create a prompt for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 17 answers starting with "A {concept_name}" to the following question: "What does a {concept_name} look like? A good example for 'dog' could be "A dog typically has four legs, a tail, and pointy ears."."""
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message},
            {"role": "assistant", "content": f"prompt: A photo of a {concept_name} in a unique environment, showcasing a variety of coat colors and patterns, featuring different expressions and poses, depicted across four seasons."},
            {"role": "user", "content": base_message}
        ]
    )

    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

cls_count_dict = PACS_count
metaprompt_json_path = './prompts/temp_base_metaprompts.json'
totalprompt_json_path = './prompts/dynamic_totalprompts_with_cot_50_PACS.json'

client = OpenAI(api_key="sk-proj-b6mF6aJroOzev4yh1afBT3BlbkFJcgqlS8S3hrxASu62u3a6")
classes = list(cls_count_dict.keys())
num_metaprompts = 10
num_prompts_per_metaprompt = 5

# -------------------------------------------------------------------
# Metaprompt generation
cls_metaprompt_dict = {}
for cls in tqdm(classes):
    metaprompts = []

    for i in tqdm(range(num_metaprompts), desc=f"Generating metaprompts for {cls}"):
        try:
            prompt = generate_prompt_stage1(client, metaprompts, cls)
            print(f"Previous prompt list: {metaprompts}")
            print(f"Generated metaprompt for stage: {prompt}")
            metaprompts.append(prompt)
        except:
            pass

    cls_metaprompt_dict[cls] = metaprompts

with open(metaprompt_json_path, 'w') as f:
    json.dump(cls_metaprompt_dict, f)


# -------------------------------------------------------------------
# # Diversified prompt generation
# # Load metaprompt pickle file
# with open(metaprompt_json_path, 'r') as f:
#     metaprompt_json_dict = json.load(f)

# totalprompt_dict = {}
# for cls in classes:
#     cls_list = []
#     metaprompts_cls = metaprompt_json_dict[cls]
#     for i, metaprompt in enumerate(tqdm(metaprompts_cls)):
#         # Generate diversified prompts for each metaprompt
#         metaprompt_dict = {'index': i, 'prompts': []}
#         diversified_prompts = []
#         for j in range(num_prompts_per_metaprompt):
#             try:
#                 prompt = generate_prompt_stage2(client, metaprompt, diversified_prompts, cls)
#                 print(f"previous generated prompts: {diversified_prompts}")
#                 print(f"Generated prompt: {prompt}")
#                 diversified_prompts.append(prompt)
#                 metaprompt_dict['prompts'].append({'index': j, 'content': prompt})
#             except:
#                 print(f"Error while generating diversified prompts for {cls}, {metaprompt}")
#         cls_list.append({metaprompt: metaprompt_dict})
#     totalprompt_dict[cls] = cls_list

# with open(totalprompt_json_path, 'w') as f:
#     json.dump(totalprompt_dict, f)