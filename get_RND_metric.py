import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import ImageDataset, get_statistics
from utils.train_utils import select_model
from torchvision import transforms
from sklearn.metrics import f1_score
import torch
import numpy as np
from utils.joint_dataset import CustomDataset


# Get Recognizability & Diversity
img_folder_dir = '/home/vision/mjlee/New_Nameonly_CL/dataset/cct/cct_train_ma'
dataset = 'cct'
type_name = 'train_ma'
model_name = 'resnet18'   # resnet18 / vit
ckpt_dir = 'dataset/RnD_ckpt'
ckpt_path = os.path.join(ckpt_dir, f'{model_name}_{dataset}.pth')
batch_size = 128

ckpt = torch.load(ckpt_path)


mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset, type_name=type_name)
model = select_model(model_name, dataset=dataset, num_classes=n_classes).cuda()
model.load_state_dict(ckpt['model_state_dict'])
exposed_classes = ckpt['exposed_classes']

test_transform = transforms.Compose(
    [
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_dataset = CustomDataset(img_folder_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


y_true, y_pred = [], []
feature_list = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        x = inputs
        y = labels
        x = x.cuda()
        y = y.cuda()
        logit, feature = model(x, get_feature=True)

        _, preds = logit.topk(1, 1, True, True)
        
        y_true.append(y.cpu())
        y_pred.append(preds.squeeze(1).cpu())
        feature_list.append(feature.cpu())

feature_list = torch.cat(feature_list)
y_true = torch.cat(y_true)
y_pred = torch.cat(y_pred)
std_list = []
for cls in range(n_classes):
    indices = [index for index, value in enumerate(y_true) if int(value) == cls]
    # print(f"cls_{cls}: {len(indices)}")
    indices = torch.tensor(indices)
    selected_features = feature_list[indices]
    # print("full", feature_list.shape)
    # print("selected", selected_features.shape)
    
    std = torch.std(selected_features)
    # print("std", std)
    std_list.append(std)
    
        
diversity = np.mean(std_list)
# print("len", len(y_true), len(y_pred))
recognizability = f1_score(y_true, y_pred, average='weighted')*100

with open(f"results/RND/{model_name}_{dataset}_{type_name}.log", 'w') as rnd_file:
    rnd_file.write(f"Diversity: {diversity}\n")
    rnd_file.write(f"Recognizability: {recognizability}")