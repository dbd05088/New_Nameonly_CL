import os
import glob

# Define the path to the folder containing your image folders
root_folder = '/home/user/mjlee/new/New_Nameonly_CL/dataset/PACS_final/PACS_final_RMD_web_temp_2'

# Walk through each subfolder in the root folder
for subdir, _, _ in os.walk(root_folder):
    # Use glob to find all files starting with '.' in each subfolder
    for file in glob.glob(os.path.join(subdir, '.*')):
        # Check if the file is an image by its extension, you can add or remove extensions according to your needs
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            os.remove(file)  # Delete the file
            print(f'Deleted: {file}')  # Optional: print out the name of the deleted file for confirmation
