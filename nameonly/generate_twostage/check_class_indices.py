import os
import argparse
from classes import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-s', '--source_dir')
args = parser.parse_args()

cls_num_dict = count_dict[args.dataset]

classes = os.listdir(args.source_dir)
cls_dir_count_dict = {cls: len(os.listdir(os.path.join(args.source_dir, cls))) for cls in classes}

cls_not_generated = []
for cls in cls_num_dict.keys():
    if cls not in cls_dir_count_dict.keys():
        cls_not_generated.append((cls, 0))
    else:
        cls_dir_count = cls_dir_count_dict[cls]
        if cls_dir_count < cls_num_dict[cls]:
            cls_not_generated.append((cls, cls_dir_count))

# Find indices
class_list = list(cls_num_dict.keys())
result_list = []
for cls, count in cls_not_generated:
    index = class_list.index(cls)
    result_list.append((f"{index}-{cls}:{count}"))

print(result_list)