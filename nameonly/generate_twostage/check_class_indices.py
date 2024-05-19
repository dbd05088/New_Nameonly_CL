import os
import argparse
from classes import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-r', '--source_dir')
parser.add_argument('-s', '--start_idx', type=int)
parser.add_argument('-e', '--end_idx', type=int)
parser.add_argument('-t', '--threshold', type=int)
parser.add_argument('--threshold_ratio', type=float)
args = parser.parse_args()

cls_num_dict = count_dict[args.dataset]

classes = os.listdir(args.source_dir)
cls_dir_count_dict = {cls: len(os.listdir(os.path.join(args.source_dir, cls))) for cls in classes
                      if os.path.isdir(os.path.join(args.source_dir, cls))}

cls_not_generated = []
for cls in list(cls_num_dict.keys()):
    if cls not in cls_dir_count_dict.keys():
        cls_not_generated.append((cls, 0))
    else:
        cls_dir_count = cls_dir_count_dict[cls]
        if args.threshold is not None:
            if cls_dir_count < args.threshold:
                cls_not_generated.append((cls, cls_dir_count))
        elif args.threshold_ratio is not None:
            if cls_dir_count < cls_num_dict[cls] * args.threshold_ratio:
                cls_not_generated.append((cls, cls_dir_count))
        elif cls_dir_count < cls_num_dict[cls]:
            cls_not_generated.append((cls, cls_dir_count))

# Find indices
class_list = list(cls_num_dict.keys())
result_list = []
for cls, count in cls_not_generated:
    index = class_list.index(cls)
    if args.start_idx is not None:
        if index < args.start_idx or index > args.end_idx:
            continue
    result_list.append((f"{index}-{cls}:{count}"))

print(result_list)