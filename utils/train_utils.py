import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import pandas as pd
from models import mnist, cifar, imagenet
from torch.utils.data import DataLoader
from onedrivedownloader import download as dn
from torch.optim import SGD
import timm
import copy

from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from tqdm import tqdm

from torchvision import models
from models.open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from transformers import AutoFeatureExtractor, ResNetModel

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        #if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )
        #print("transform")
        #print(self.transforms)
        x_out = self.transforms(x)  # BxCxHxW
        '''
        else:
            additional_aug = self.randaugmentation.form_transforms(list(set((labels))))
            
            self.before_transforms = nn.Sequential(
                K.Resize(size = (self.inp_size, self.inp_size)),
                K.RandomCrop(size = (self.inp_size, self.inp_size)),
                K.RandomHorizontalFlip(p=1.0)
                )
            x_out = self.before_transforms(x)
            
            for i in range(len(x)):
                add_transform = nn.Sequential(*additional_aug[labels[i]])
                x_out[i] = add_transform(x_out[i])
                
            self.after_transforms = nn.Sequential(
                K.Normalize(self.mean, self.std)
                )
            x_out = self.transforms(x_out)  # BxCxHxW
        '''
        ##### check transforms
        # print("self.transform")
        # print(self.transforms)

        #x_out, _ = self.cutmix(x_out)
        return x_out


def get_transform(dataset, transform_list, gpu_transform, use_kornia=True):
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset)
    if use_kornia:
        train_transform = DataAugmentation(inp_size, mean, std)
    else:
        train_transform = []
        if "cutout" in transform_list:
            train_transform.append(Cutout(size=16))
            if gpu_transform:
                gpu_transform = False
                print("cutout not supported on GPU!")
        if "randaug" in transform_list:
            train_transform.append(transforms.RandAugment())
            
        if "autoaug" in transform_list:
            if hasattr(transform_list, 'AutoAugment'):
                if 'cifar' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
                elif 'imagenet' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            else:
                train_transform.append(select_autoaugment(dataset))
                gpu_transform = False
        if "trivaug" in transform_list:
            train_transform.append(transforms.TrivialAugmentWide())
        if gpu_transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform

def select_optimizer(opt_name, lr, model):
    if hasattr(model, 'fc'):
        fc_name = 'fc'
    elif hasattr(model, 'head'):
        fc_name = 'head'
    if "adam" in opt_name:
        params = [param for name, param in model.named_parameters() if fc_name not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif "sgd" in opt_name:
        params = [param for name, param in model.named_parameters() if fc_name not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    if 'freeze_fc' not in opt_name:
        opt.add_param_group({'params': getattr(model, fc_name).parameters()})
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler



def get_ckpt_remote_url(pre_dataset):
    if pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"

    elif pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"

    elif pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

    else:
        raise ValueError("Unknown auxiliary dataset")


def load_initial_checkpoint(pre_dataset, model, device, load_cp_path = None):
    url, ckpt_name = get_ckpt_remote_url(pre_dataset)
    load_cp_path = load_cp_path if load_cp_path is not None else './checkpoints/'
    print("Downloading checkpoint file...")
    dn(url, load_cp_path)
    print(f"Downloaded in: {load_cp}")
    net = load_cp(load_cp_path, model, device, moco=True)
    print("Loaded!")
    return net

def generate_initial_checkpoint(net, pre_dataset, pre_epochs, num_aux_classes, device, opt_args):
    aux_dset, aux_test_dset = get_aux_dataset()
    net.fc = torch.nn.Linear(net.fc.in_features, num_aux_classes).to(device)
    net.train()
    opt = SGD(net.parameters(), lr=opt_args["lr"], weight_decay=opt_args["optim_wd"], momentum=opt_args["optim_mom"])
    sched = None
    if self.args.pre_dataset.startswith('cub'):
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[80, 150, 250], gamma=0.5)
    elif 'tinyimg' in self.args.pre_dataset.lower():
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[20, 30, 40, 45], gamma=0.5)

    for e in range(pre_epochs):
        for i, (x, y, _) in tqdm(enumerate(aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(aux_dl)):
            y = y.long()
            opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            aux_out = net(x)
            aux_loss = loss(aux_out, y)
            aux_loss.backward()
            opt.step()

        if sched is not None:
            sched.step()
        if e % 5 == 4:
            print(e, f"{self.mini_eval()*100:.2f}%")
    from datetime import datetime
    # savwe the model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelpath = "my_checkpoint" + '_' + now + '.pth'
    torch.save(net.state_dict(), modelpath)
    print(modelpath)

def load_cp(cp_path, net, device, moco=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    print("net")
    print([name for name, _ in net.named_parameters()])
    s = torch.load(cp_path, map_location=device)
    print("s keys", s.keys())
    '''
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
             i in s['state_dict'].items() if 'encoder_q' in k}
    '''

    #if not ignore_classifier: # online CL이므로 fc out-dim을 1부터 시작
    net.fc = torch.nn.Linear(
        net.fc.in_features, 1).to(device) # online이므로 num_aux_classes => 1

    for k in list(s):
        if 'fc' in k:
            s.pop(k)
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
    except:
        _, unm = net.load_state_dict(s, strict=False)
        print("unm")
        print(unm)
        '''
        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"
        '''

    return net
'''
def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(self.device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(self.device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps
'''
def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )

    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def select_model(model_name, dataset, num_classes=None, opt_dict=None, G=False, F=False, ver2=False):
    if model_name == 'vit':
        if G:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            # layer_name = 'blocks.9.norm1'
            # hook_model = ViTModel(base_model, layer_name, G=G)
            # layers = list(model.children())[:4]
            # layers.extend(list(model.children())[4][:9])
            # print("param list", list(model.children()))
            # print("length", len(list(model.children())))
            layers=[model.patch_embed]
            layers.extend(model.blocks[:8])
            # model.cls_token, model.pos_embed, nn.Sequential(*layers)
            return nn.Sequential(*layers)
        elif F:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            layers = model.blocks[8:]
            layers.append(model.head)
            
            return model.norm, nn.Sequential(*layers)
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            return ViTModel(model)
    
    elif model_name == "clip_resnet":
        model, preprocess_train, preprocess_val = create_model_and_transforms('RN50', pretrained="yfcc15m")
        tokenizer = get_tokenizer('RN50')
        criterion = create_loss()
        return model, preprocess_train, preprocess_val, tokenizer, criterion
    elif model_name == "clip_vit": 
        model, preprocess_train, preprocess_val = create_model_and_transforms('ViT-B-32', pretrained="openai")
        tokenizer = get_tokenizer('ViT-B-32')
        criterion = create_loss()
        return model, preprocess_train, preprocess_val, tokenizer, criterion

    elif model_name == "resnet50":
        # feature_extractor = AutoFeatureExtractor.from_pretrained('Ramos-Ramos/dino-resnet-50')
        model = ResNetModel.from_pretrained('Ramos-Ramos/dino-resnet-50')
        # model = MultiCropWrapper(model, DINOHead(
        #     model.embed_dim,
        #     num_classes,
        # ))
        model = ResNet50(model, num_classes=num_classes)
        return model
        
    
    model_imagenet = False
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

   
    
    #model_class = getattr(imagenet, "ResNet")
    model_imagenet=True
    model_class = getattr(cifar, "ResNet")
    if G:
        model_class = getattr(cifar, "ResNet_G")
        opt["ver2"] = ver2
    elif F:
        model_class = getattr(cifar, "ResNet_F")
        opt["ver2"] = ver2
    
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt, model_imagenet)

    # TODO initial check
    initial = False

    # pre_dataset 설정
    pre_dataset = None
    pre_dataset_num_class = 0
    path_load_cp = None
    if dataset == "cifar10":
        pre_dataset = "cifar100"
        path_load_cp = "res18_cifar100_pretrained_model.pth" #"checkpoint/rs18_cifar100_new.pth"
        pre_dataset_num_class = 100
    elif dataset == "cifar100":
        pre_dataset = "tiny_imagenet"
        pre_dataset_num_class = 1000
    else:
        pre_dataset = "anything"

    assert pre_dataset is not None # none이면 설정이 덜 된것
    
    return model

##### for ASER #####
def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


#### For x_der ###
def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)


def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


class strong_aug():
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.ToTensor()
        ])
        self.mean = mean
        self.std = std

    def __call__(self, x):
        flip = random_flip(x)
        tmp = torch.stack(
                [self.transform(a) for a in flip]
            )
        tmp2 = random_grayscale(
            tmp)
        y = normalize(tmp2, self.mean, self.std)
        return y


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean(0)

        return loss.mean() if self.reduction == 'mean' else loss.sum()
    
    
class ViTModel(nn.Module):
    def __init__(self, vit_model, layer_name=None, G=False):
        super(ViTModel, self).__init__()
        self.vit_model = vit_model
        self.layer_name = layer_name
        # self.intermediate_output = None
        # self.hook_handle = self.vit_model.norm.register_forward_hook(self.hook_fn)
        self.head = copy.deepcopy(self.vit_model.head)
        self.vit_model.head = nn.Identity()
        self.G = G

    def hook_fn(self, module, input, output):
        print("module name", module._get_name())
        self.intermediate_output = output

    def forward(self, x, get_feature=False):
        features = self.vit_model(x) 
        output = self.head(features)
        if get_feature:
            return output, features
        return output
    
class ResNet50(nn.Module):
    def __init__(self, resnet_model, num_classes=None, layer_name=None, G=False):
        super(ResNet50, self).__init__()
        self.encoder = resnet_model
        self.fc = nn.Linear(100352, num_classes)

    def forward(self, x, get_feature=False):
        # inputs = self.feature_extractor(x)
        features = self.encoder(x) 
        features = features.last_hidden_state
        features = features.view(x.size(0), -1)
        output = self.fc(features)
        if get_feature:
            return output, features
        return output

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
