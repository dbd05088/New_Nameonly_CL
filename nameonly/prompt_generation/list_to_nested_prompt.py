# Nested prompt structure -> list of prompts

import json
from classes import *

prompt_path = './prompts/gpt4_wo_cot_wo_hierarchy.json'
prompt_list = [
    "A photo of [concept] in bright, vivid colors.",
    "A photo of [concept] in soft pastel shades.",
    "A photo of [concept] at sunset with golden hues.",
    "A photo of [concept] in monochrome.",
    "A photo of [concept] in a foggy landscape.",
    "A photo of [concept] in vibrant urban settings.",
    "A photo of [concept] in a lush green forest.",
    "A photo of [concept] by a serene lake.",
    "A photo of [concept] in a bustling market.",
    "A photo of [concept] in a snowy mountain scene.",
    "A photo of [concept] in autumnal colors.",
    "A photo of [concept] in a minimalist setting.",
    "A photo of [concept] in a desert with warm tones.",
    "A photo of [concept] in a busy street at night.",
    "A photo of [concept] in a colorful garden.",
    "A photo of [concept] with a dark, moody atmosphere.",
    "A photo of [concept] in a rustic farmhouse.",
    "A photo of [concept] in a modern, sleek environment.",
    "A photo of [concept] under the starry night sky.",
    "A photo of [concept] in a historical setting.",
    "A photo of [concept] in a tropical paradise.",
    "A photo of [concept] in a quiet library.",
    "A photo of [concept] in a vibrant festival.",
    "A photo of [concept] in a serene beach scene.",
    "A photo of [concept] in a futuristic cityscape.",
    "A photo of [concept] in an industrial setting.",
    "A photo of [concept] in a picturesque village.",
    "A photo of [concept] in a magical, enchanted forest.",
    "A photo of [concept] during a dramatic storm.",
    "A photo of [concept] in a cozy, warm cabin.",
    "A photo of [concept] in a sprawling countryside.",
    "A photo of [concept] in a vibrant, neon-lit street.",
    "A photo of [concept] in a tranquil meadow.",
    "A photo of [concept] with an antique, vintage look.",
    "A photo of [concept] in a high-tech laboratory.",
    "A photo of [concept] in a grand, ornate palace.",
    "A photo of [concept] on a busy train station platform.",
    "A photo of [concept] in a mystical, fog-covered swamp.",
    "A photo of [concept] in a sunny park.",
    "A photo of [concept] in a quiet, snowy village.",
    "A photo of [concept] in a vibrant street market.",
    "A photo of [concept] in an underwater scene.",
    "A photo of [concept] in a colorful street art alley.",
    "A photo of [concept] in a serene Zen garden.",
    "A photo of [concept] in a dramatic mountain landscape.",
    "A photo of [concept] in a sunlit attic.",
    "A photo of [concept] on a busy city sidewalk.",
    "A photo of [concept] in an old, abandoned building.",
    "A photo of [concept] in a tropical rainforest.",
    "A photo of [concept] in a high-energy sports event."
]

prompt_result_dict = {"metaprompts": [
    {
        "index": 0,
        "metaprompt": "dummy",
        "prompts": []
    }
]}

for i, prompt in enumerate(prompt_list):
    prompt_result_dict["metaprompts"][0]['prompts'].append(
        {
            "index": i,
            "content": prompt
        }
    )

with open(prompt_path, 'w') as f:
    json.dump(prompt_result_dict, f)

