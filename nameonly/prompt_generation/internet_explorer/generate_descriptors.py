import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(target_dir)
from classes import get_count_dict
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

prompt = """What are some words that describe the quality of 'motorcycle'? 
The motorcycle is frail. 
The motorcycle is red. 
The motorcycle is humongous. 
The motorcycle is tall. 
The motorcycle is"""

def generate_descriptors(concept):
    prompt = f"""What are some words that describe the quality of '{concept}'?
    The {concept} is frail.
    The {concept} is red.
    The {concept} is humongous.
    The {concept} is tall.
    The {concept} is"""
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
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
        
    return descriptors


breakpoint()
# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate output
outputs = model.generate(
    input_ids,
    max_length=100,
    temperature=0.9,
    num_return_sequences=5,
    do_sample=True
)

# 결과 출력
descriptors = [tokenizer.decode(output, skip_special_tokens=True).split()[-1] for output in outputs]
print(descriptors)