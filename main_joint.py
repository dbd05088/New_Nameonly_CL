import os
import torch
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms, models
from utils.joint_dataset import CustomDataset
from utils.train_utils import select_model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_name', type=str, default='resnet18', help='Model name')
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args()

# Load dataset
if args.dataset == 'PACS_final':
    domain_dirs = {
        'id': './dataset/PACS_final/PACS_final_test_ma',
        'art_painting': './dataset/PACS_final/PACS_MA/art_painting',
        'cartoon': './dataset/PACS_final/PACS_MA/cartoon',
        'sketch': './dataset/PACS_final/PACS_MA/sketch'
    }
elif args.dataset == 'DomainNet':
    domain_dirs = {
        'id': './dataset/DomainNet/DomainNet_test_ma',
        'clipart': './dataset/DomainNet/DomainNet_MA/clipart',
        'infograph': './dataset/DomainNet/DomainNet_MA/infograph',
        'painting': './dataset/DomainNet/DomainNet_MA/painting',
        'quickdraw': './dataset/DomainNet/DomainNet_MA/quickdraw',
        'sketch': './dataset/DomainNet/DomainNet_MA/sketch',
    }
elif args.dataset == 'NICO':
    domain_dirs = {
        'id': None,
        'autumn': './dataset/NICO/NICO_MA/autumn',
        'dim': './dataset/NICO/NICO_MA/dim',
        'grass': './dataset/NICO/NICO_MA/grass',
        'outdoor': './dataset/NICO/NICO_MA/outdoor',
        'rock': './dataset/NICO/NICO_MA/rock',
        'water': './dataset/NICO/NICO_MA/water'
    }
elif args.dataset == 'cct':
    domain_dirs = {
        'id': './dataset/cct/cct_in_test_ma',
        'out': './dataset/cct/cct_out_test_ma'
    }
elif args.dataset == 'cifar10':
    domain_dirs = {
        'id': './dataset/cifar10/cifar10_original_test',
        'c': './dataset/cifar10/cifar10_c',
        'nc': './dataset/cifar10/cifar10_nc'
    }

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_root_dir = os.path.join('./dataset', args.dataset, f'{args.dataset}_{args.type}')
train_dataset = CustomDataset(train_root_dir, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_datasets = {}
for domain, domain_dir in domain_dirs.items():
    if domain_dir is not None:
        test_dataset = CustomDataset(domain_dir, transform=transform)
        test_datasets[domain] = test_dataset
        
        # Get the number of classes
        num_classes = len(set(train_dataset.labels))
    else:
        test_datasets[domain] = None
test_dataloaders = {domain: torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) for domain, test_dataset in test_datasets.items()}


# Training
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, criterion, test_dataloader, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
    return running_loss / len(test_dataloader), corrects / total
                
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = select_model(args.model_name, dataset=args.dataset, num_classes=num_classes).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    eval_results = {
        'in_domain_intermediate': [],
        'ood_domain_intermediate': {}
    }
    
    # Create csv save directory
    save_dir = os.path.join('./results', args.dataset, "joint", args.type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_path = os.path.join(save_dir, f'{args.model_name}.csv')
    with open(csv_path, 'w') as f:
        f.write('Epoch,In Domain,OOD\n')
    
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}")
        train_loss = train(model, criterion, optimizer, train_dataloader, device)
        # Evaluate
        for domain, test_dataloader in tqdm(test_dataloaders.items()):
            if test_dataloader is not None:
                test_loss, accuracy = test(model, criterion, test_dataloader, device)
                if domain == 'id':
                    eval_results['in_domain_intermediate'].append(accuracy)
                else:
                    if domain not in eval_results['ood_domain_intermediate']:
                        eval_results['ood_domain_intermediate'][domain] = []
                    eval_results['ood_domain_intermediate'][domain].append(accuracy)
        
        # Print intermediate results
        ood_average_current = np.mean(
            [eval_results['ood_domain_intermediate'][domain][-1] for domain in eval_results['ood_domain_intermediate']],
            axis=0
        )
        print(f"Epoch {epoch+1}/{args.num_epochs}, In Domain Accuracy: {eval_results['in_domain_intermediate'][-1]:.4f}")
        print(f"Epoch {epoch+1}/{args.num_epochs}, OOD Domain Accuracy: {ood_average_current:.4f}")

        # Save intermediate results
        epoch_data = {
            'Epoch': epoch,
            'In Domain': eval_results['in_domain_intermediate'][-1],
            'OOD': ood_average_current
        }
        df_epoch = pd.DataFrame(epoch_data, index=[0])
        df_epoch.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Save average results
    in_domain_average = np.mean(eval_results['in_domain_intermediate'], axis=0)
    ood_average = np.mean(
        [eval_results['ood_domain_intermediate'][domain] for domain in eval_results['ood_domain_intermediate']],
        axis=0
    )
    ood_average = np.mean(ood_average)
    df_average = pd.DataFrame({
        'Epoch': 'Average',
        'In Domain': in_domain_average,
        'OOD': ood_average
    }, index=[0])
    df_average.to_csv(csv_path, mode='a', header=False, index=False)
    
if __name__ == '__main__':
    main()