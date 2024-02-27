import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

def get_stat(image_root_dir):
    # Calculate mean and std for each channel

    # Get all images recursively
    images = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith(tuple(image_exts)):
                images.append(os.path.join(root, file))

    print(f"Found {len(images)} images")
    # Calculate mean and std
    means, stds = [], []

    # Read all images and calculate mean and std
    for image in tqdm(images):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        means.append(np.mean(img / 256, axis=(0, 1)))
        stds.append(np.std(img / 256, axis=(0, 1)))

    # Calculate mean and std for all images
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return {'mean': mean, 'std': std}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mean and std for images')
    parser.add_argument('--image_root_dir', type=str, help='Root directory of images')
    args = parser.parse_args()
    result = get_stat(args.image_root_dir)

    print(f"Mean: {result['mean']}")
    print(f"Std: {result['std']}")
