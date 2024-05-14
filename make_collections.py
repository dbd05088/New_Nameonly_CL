import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import json

def make_collections(dir, seed, dataset, cls_to_tasks, class_to_dict=None):
    dir = os.path.normpath(dir)
    class_lists = os.listdir(dir)
    base_dir = os.path.basename(os.path.normpath(dir))
    sample_streams = []
    if class_to_dict is None:
        np.random.seed(seed)
        np.random.shuffle(class_lists)
        class_to_dict = {label : idx for idx, label in enumerate(class_lists)}
    else:
        class_lists = list(class_to_dict.keys())
    times = np.linspace(0, 1, len(class_lists))
    task_streams_count = []
    for idx, num_cls in enumerate(cls_to_tasks):
        start_idx = sum(cls_to_tasks[:idx]) if idx != 0 else 0
        task_streams = []
        for cls in class_lists[start_idx:start_idx + num_cls]:
            file_lists = os.listdir(os.path.join(dir, cls))
            for f in file_lists:
                dic = {}
                dic["file_name"] = os.path.join(base_dir, cls, f)
                dic["klass"] = cls
                dic["label"] = class_to_dict[cls]
                dic["time"] = times[idx]
                task_streams.append(dic)
        print("task", idx, class_lists[start_idx:start_idx + num_cls], len(task_streams))
        task_streams_count.append(len(task_streams))
        random.shuffle(task_streams)
        sample_streams.extend(task_streams)
    json_dict = {}
    json_dict["stream"] = sample_streams
    json_dict["cls_dict"] = class_to_dict
    json_dict["cls_addition"] = np.zeros(len(class_lists)).tolist()
    type_name = dir.split("/")[-1]

    if "_" in dataset:
        new_type_name =  "_".join(type_name.split("_")[2:])
    else:
        new_type_name = "_".join(type_name.split("_")[1:])
    
    file_path = f"{dataset}_{new_type_name}_sigma0_repeat1_init100_seed{seed}.json"
    print(file_path)
    with open(file_path, 'w') as outfile:
        json.dump(json_dict, outfile)
    print(task_streams_count)
    print([sum(task_streams_count[:idx+1]) for idx in range(len(task_streams_count))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mean and std for images')
    parser.add_argument('-r', '--root_dir', type=str, help='Root directory of images')
    parser.add_argument('-s', '--random_seed', help='random seed')
    parser.add_argument('-d', '--dataset', type=str, help='name of the dataset')
    parser.add_argument('-c', '--cls_to_tasks', type=str, help='number of classes per task')
    args = parser.parse_args()
    cls_to_tasks = [int(num) for num in args.cls_to_tasks.split()]
    seeds = [int(s) for s in args.random_seed.split()]
    print(cls_to_tasks)
    print(f"Seeds: {seeds}")
    class_to_dict = None

    if args.dataset == "PACS_final":
        class_to_dict_mappings = {
            1: {'dog':0, 'horse':1, 'guitar':2, 'elephant':3, 'giraffe':4, 'person':5, 'house':6},
            2: {'giraffe':0, 'guitar':1, 'person':2, 'horse':3, 'dog':4, 'house':5, 'elephant':6},
            3: {'giraffe':0, 'dog':1, 'house':2, 'person':3, 'guitar':4, 'elephant':5, 'horse':6},
            4: {'giraffe':0, 'dog':1, 'person':2, 'elephant':3, 'guitar':4, 'house':5, 'horse':6},
            5: {'dog':0, 'horse':1, 'giraffe':2, 'guitar':3, 'elephant':4, 'house':5, 'person':6}
        }
    elif args.dataset == "cct":
        class_to_dict_mappings = {
            1: {'rabbit': 0, 'rodent': 1, 'car': 2, 'bird': 3, 'skunk': 4, 'bobcat': 5, 'squirrel': 6, 'coyote': 7, 'dog': 8, 'opossum': 9, 'cat': 10, 'raccoon': 11},
            2: {'opossum': 0, 'raccoon': 1, 'car': 2, 'coyote': 3, 'skunk': 4, 'bird': 5, 'squirrel': 6, 'rodent': 7, 'rabbit': 8, 'bobcat': 9, 'dog': 10, 'cat': 11},
            3: {'raccoon': 0, 'car': 1, 'skunk': 2, 'rabbit': 3, 'dog': 4, 'bobcat': 5, 'coyote': 6, 'squirrel': 7, 'rodent': 8, 'opossum': 9, 'cat': 10, 'bird': 11},
            4: {'rodent': 0, 'car': 1, 'bobcat': 2, 'opossum': 3, 'cat': 4, 'rabbit': 5, 'dog': 6, 'squirrel': 7, 'skunk': 8, 'raccoon': 9, 'coyote': 10, 'bird': 11},
            5: {'coyote': 0, 'raccoon': 1, 'rabbit': 2, 'opossum': 3, 'dog': 4, 'car': 5, 'cat': 6, 'skunk': 7, 'squirrel': 8, 'bird': 9, 'bobcat': 10, 'rodent': 11}
        }
    # if args.dataset == "PACS":
    #     if args.random_seed == 1:
    #         class_to_dict = {"dog": 0, "horse": 1, "house": 2, "person": 3, "giraffe": 4, "guitar": 5, "elephant": 6}
    #     elif args.random_seed == 2:
    #         class_to_dict = {"giraffe": 0, "house": 1, "guitar": 2, "horse": 3, "dog": 4, "elephant": 5, "person": 6}
    #     elif args.random_seed == 3:
    #         class_to_dict = {"giraffe": 0, "dog": 1, "elephant": 2, "guitar": 3, "house": 4, "person": 5, "horse": 6}
    #     elif args.random_seed == 4:
    #         class_to_dict = {"giraffe": 0, "dog": 1, "guitar": 2, "person": 3, "house": 4, "elephant": 5, "horse": 6}
    #     elif args.random_seed == 5:
    #         class_to_dict = {"dog": 0, "horse": 1, "giraffe": 2, "house": 3, "person": 4, "elephant": 5, "guitar": 6}


    # elif args.dataset == "PACS_final":
    #     if args.random_seed == 1:
    #         class_to_dict = {'dog':0, 'horse':1, 'guitar':2, 'elephant':3, 'giraffe':4, 'person':5, 'house':6}
    #     elif args.random_seed == 2:
    #         class_to_dict = {'giraffe':0, 'guitar':1, 'person':2, 'horse':3, 'dog':4, 'house':5, 'elephant':6}
    #     elif args.random_seed == 3:
    #         class_to_dict = {'giraffe':0, 'dog':1, 'house':2, 'person':3, 'guitar':4, 'elephant':5, 'horse':6}
    #     elif args.random_seed == 4:
    #         class_to_dict = {'giraffe':0, 'dog':1, 'person':2, 'elephant':3, 'guitar':4, 'house':5, 'horse':6}
    #     elif args.random_seed == 5:
    #         class_to_dict = {'dog':0, 'horse':1, 'giraffe':2, 'guitar':3, 'elephant':4, 'house':5, 'person':6}

    # elif args.dataset == "cifar10":
    #     if args.random_seed == 1:
    #         class_to_dict = {"ship": 0, "dog": 1, "frog": 2, "automobile": 3, "bird": 4, "horse": 5, "cat": 6, "truck": 7, "airplane": 8, "deer": 9}
    #     elif args.random_seed == 2:
    #         class_to_dict = {"automobile": 0, "cat": 1, "deer": 2, "bird": 3, "truck": 4, "ship": 5, "horse": 6, "frog": 7, "dog": 8, "airplane": 9}
    #     elif args.random_seed == 3:
    #         class_to_dict = {"deer": 0, "automobile": 1, "cat": 2, "ship": 3, "dog": 4, "frog": 5, "truck": 6, "bird": 7, "horse": 8, "airplane": 9}
    #     elif args.random_seed == 4:
    #         class_to_dict = {"horse": 0, "airplane": 1, "automobile": 2, "dog": 3, "ship": 4, "frog": 5, "bird": 6, "cat": 7, "deer": 8, "truck": 9}
    #     elif args.random_seed == 5:
    #         class_to_dict = {"dog": 0, "deer": 1, "ship": 2, "automobile": 3, "truck": 4, "cat": 5, "bird": 6, "airplane": 7, "frog": 8, "horse": 9}

    for seed in seeds:
        result = make_collections(args.root_dir, seed, args.dataset, cls_to_tasks, class_to_dict = class_to_dict_mappings[seed])
    # result = make_collections(args.root_dir, args.random_seed, args.dataset, cls_to_tasks, class_to_dict = class_to_dict)

