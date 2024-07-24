from img2dataset import download
import shutil
import os
import argparse
from classes import *
from tqdm import tqdm

url_path = './urls/Bongard_flickr'
url_list = os.listdir(url_path)
for url_file in tqdm(url_list):
    id = url_file.split('.')[0]
    output_dir = os.path.join('datasets', f"Bongard_flickr", id)
    print(f"Downloading {id} to {output_dir}")
    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    download(
        processes_count=1,
        thread_count=1,
        url_list=f"./urls/Bongard_flickr/{id}.txt",
        image_size=256,
        resize_mode="keep_ratio",
        output_folder=output_dir,
        output_format="files",
    )
