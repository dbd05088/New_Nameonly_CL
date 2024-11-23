import os
import sys
import json
from tqdm import tqdm
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(target_dir)
from classes import get_count_dict

def generate_descriptors_with_openai(client, concept):
    """
    """
    prompt = f"""What are some words that describe the quality of '{concept}'?
    The {concept} is frail.
    The {concept} is red.
    The {concept} is humongous.
    The {concept} is tall.
    The {concept} is"""

    try:
        # Generate one prompt using GPT-4 API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are an assistant that generates descriptive words for a given concept."},
                    {"role": "user", "content": prompt},
                ]
        )
        
        response_content = response.choices[0].message.content
        processed_descriptors = []
        for desc in response_content.split("\n"):
            desc = desc.split(",")[0].split(".")[0].strip()
            if desc.lower() != concept.lower():
                processed_descriptors.append(desc)

        # Remove the part "The {concept} is" from the descriptors if it exists
        processed_descriptors = [desc.replace(f"The {concept} is ", "") for desc in processed_descriptors]
        processed_descriptors = [desc.replace(f"- ", "") for desc in processed_descriptors]
        return processed_descriptors

    except Exception as e:
        print(f"Error generating descriptors for '{concept}': {e}")
        return []

# Get concept count dictionary
client = OpenAI(api_key="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")
count_dict = get_count_dict("PACS")
result_json_path = "PACS_descriptors.json"
concepts = list(count_dict.keys())
descriptors = {}

# Generate descriptors for each concept
for concept in tqdm(concepts):
    unique_descriptors = set()
    while len(unique_descriptors) < 32:
        unique_descriptors.update(generate_descriptors_with_openai(client, concept))
        print(f"Unique descriptors for '{concept}': {unique_descriptors}")

    # Sample 32 descriptors
    unique_descriptors = list(unique_descriptors)[:32]
    descriptors[concept] = list(unique_descriptors)

# Save descriptors to a JSON file
with open(result_json_path, 'w') as f:
    json.dump(descriptors, f)
