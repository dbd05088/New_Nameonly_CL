import os
import json
source_path = '/home/user/seongwon/New_Nameonly_CL/dataset/NICO/NICO_MA/water'
target_path = '/home/user/seongwon/New_Nameonly_CL/collections/NICO/test/NICO_water_sigma0_init100_seed1.json'

if not os.path.exists(os.path.dirname(target_path)):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

classes = os.listdir(source_path)
result_list = []
base_path = os.path.join(source_path.split('/')[-2], source_path.split('/')[-1])
for cls in classes:
    cls_path = os.path.join(source_path, cls)
    images = os.listdir(cls_path)
    for image in images:
        image_dict = {
            "file_name": os.path.join(base_path, image),
            "klass": cls,
            "label": 0,
            "time": 0
        }
        result_list.append(image_dict)

with open(target_path, 'w') as f:
    json.dump(result_list, f)
