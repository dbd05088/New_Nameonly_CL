import os
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(target_dir)
from classes import get_count_dict
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model.to("cuda")

def generate_descriptors(concept):
    prompt = f"""What are some words that describe the quality of '{concept}'?
    The {concept} is frail.
    The {concept} is red.
    The {concept} is humongous.
    The {concept} is tall.
    The {concept} is"""
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=100,
        temperature=0.9,
        num_return_sequences=5,
        do_sample=True
    )
    descriptors = [tokenizer.decode(output, skip_special_tokens=True).split()[-1] for output in outputs]
    processed_descriptors = []
    for desc in descriptors:
        desc = desc.split(",")[0].split(".")[0].strip()
        if desc.lower() != concept.lower():
            processed_descriptors.append(desc)

    print(f"Descriptors for '{concept}': {processed_descriptors}")
    return processed_descriptors

count_dict = get_count_dict("PACS")
concepts = list(count_dict.keys())
descriptors = {}

for concept in concepts:
    unique_descriptors = set()
    while len(unique_descriptors) < 32:
        unique_descriptors.update(generate_descriptors(concept))
        print(f"Unique descriptors for '{concept}': {unique_descriptors}")
    descriptors[concept] = list(unique_descriptors)

    breakpoint()