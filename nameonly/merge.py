import os
import argparse
import shutil

# This script merges two directories of images into one directory.
# Images of same name will be renamed.

image_extensions = ['jpg', 'jpeg', 'png', "JPEG"]

def merge_images(root_dir1, root_dir2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_idx = 0
    # Copy images from root_dir1 to output_dir
    images = os.listdir(root_dir1)
    for image in images:
        if image.split('.')[-1] not in image_extensions:
            continue
        image_path = os.path.join(root_dir1, image)
        output_image_path = os.path.join(output_dir, str(image_idx).zfill(6) + '.png')
        if os.path.exists(output_image_path):
            # Increment image_idx until a new name is found
            while True:
                image_idx += 1
                output_image_path = os.path.join(output_dir, str(image_idx).zfill(6) + '.png')
                if not os.path.exists(output_image_path):
                    break
        shutil.copy(image_path, output_image_path)
        image_idx += 1
    
    # Copy images from root_dir2 to output_dir
    images = os.listdir(root_dir2)
    for image in images:
        if image.split('.')[-1] not in image_extensions:
            continue
        image_path = os.path.join(root_dir2, image)
        output_image_path = os.path.join(output_dir, str(image_idx).zfill(6) + '.png')
        if os.path.exists(output_image_path):
            # Increment image_idx until a new name is found
            while True:
                image_idx += 1
                output_image_path = os.path.join(output_dir, str(image_idx).zfill(6) + '.png')
                if not os.path.exists(output_image_path):
                    break
        shutil.copy(image_path, output_image_path)
        image_idx += 1
    
    print(f"Total images: {image_idx}")
    
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--root_dir1", type=str, required=True)
parser.add_argument("-b", "--root_dir2", type=str, required=True)
parser.add_argument("-c", "--target_dir", type=str, required=True)

args = parser.parse_args()

if __name__ == "__main__":
    merge_images(args.root_dir1, args.root_dir2, args.target_dir)

