import os
import time
from tqdm import tqdm

# Initialize gdrive
os.system("./gdrive account remove dbd05088@naver.com")
os.system("./gdrive account import gdrive_export-dbd05088_naver_com.tar")
os.system("./gdrive account switch dbd05088@naver.com")

# Files to upload
dataset = "PACS_final"
create_tar = True
files = [
    "PACS_final_DINO_small_moderate_filtered",
    "PACS_final_DINO_base_moderate_filtered",
    "PACS_final_CLIP_moderate_filtered",
    "PACS_final_DINO_base_CurvMatch_10_0_0001",
    "PACS_final_DINO_base_CurvMatch_25_0_0001",
    "PACS_final_DINO_base_CurvMatch_50_0_0001",
    "PACS_final_DINO_base_Glister_10_0_0001",
    "PACS_final_DINO_base_Glister_25_0_0001",
    "PACS_final_DINO_base_Glister_50_0_0001",
    "PACS_final_DINO_base_GradMatch_10_0_0001",
    "PACS_final_DINO_base_GradMatch_25_0_0001",
    "PACS_final_DINO_base_GradMatch_50_0_0001",
    "PACS_final_DINO_base_Uncertainty_10_0_0001",
    "PACS_final_DINO_base_Uncertainty_25_0_0001",
    "PACS_final_DINO_base_Uncertainty_50_0_0001",
    "PACS_final_DINO_small_CurvMatch_10_0_0001",
    "PACS_final_DINO_small_CurvMatch_25_0_0001",
    "PACS_final_DINO_small_CurvMatch_50_0_0001",
    "PACS_final_DINO_small_Glister_10_0_0001",
    "PACS_final_DINO_small_Glister_25_0_0001",
    "PACS_final_DINO_small_Glister_50_0_0001",
    "PACS_final_DINO_small_GradMatch_10_0_0001",
    "PACS_final_DINO_small_GradMatch_25_0_0001",
    "PACS_final_DINO_small_GradMatch_50_0_0001",
    "PACS_final_DINO_small_Uncertainty_10_0_0001",
    "PACS_final_DINO_small_Uncertainty_25_0_0001",
    "PACS_final_DINO_small_Uncertainty_50_0_0001",
]

# Change path
files = [os.path.join(dataset, file) for file in files]

# Create tar files if needed
if create_tar:
    for file in files:
        print(f"Creating tar file for {file}")
        breakpoint()
        os.system(f"tar -cf {file}.tar -C {dataset} {os.path.basename(file)}")

# Upload files
for file in tqdm(files):
    print(f"Uploading {file}.tar")
    result_str = os.popen(f'./gdrive files upload {file}.tar | grep "Id: "').read()
    file_id = result_str.split("Id: ")[1].strip()
    
    # Append file id to result.txt file
    with open('result.txt', 'a') as f:
        f.write(f"./gdrive files download {file_id} # {os.path.basename(file)}\n")

# Write a script to untar files
for file in tqdm(files):
    with open('result.txt', 'a') as f:
        f.write(f"tar -xf {os.path.basename(file)}.tar -C {dataset}\n")

# Write a script to remove tar files
for file in tqdm(files):
    with open('result.txt', 'a') as f:
        f.write(f"rm {os.path.basename(file)}.tar\n")