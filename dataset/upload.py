import os
import time
from tqdm import tqdm

# Initialize gdrive
os.system("./gdrive account remove dbd05088@naver.com")
os.system("./gdrive account import gdrive_export-dbd05088_naver_com.tar")
os.system("./gdrive account switch dbd05088@naver.com")

# Files to upload
dataset = "DomainNet"
create_tar = True
files = [
    "DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver6",
    "DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver7",
]
# Change path
files = [os.path.join(dataset, file) for file in files]

# Create tar files if needed
if create_tar:
    for file in files:
        print(f"Creating tar file for {file}")
        os.system(f"tar -cf {file}.tar -C {dataset} {os.path.basename(file)}")

# Remove a script text file if it exists
if os.path.exists('result.txt'):
    os.remove('result.txt')
    
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
        
# Write a script to generate statistics
with open('result.txt', 'a') as f:
    # python get_stats.py -r ./dataset/DomainNet/DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow
    for file in tqdm(files):
        f.write(f"python get_stats.py -r ./dataset/{file}\n")

# Write a script to generate json files
with open('result.txt', 'a') as f:
    for file in tqdm(files):
        f.write(f"python make_collections.py -r ./dataset/{file}\n")