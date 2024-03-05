## the code is mostly taken from autoaugment pytorch repo:
# https://github.com/DeepVoltaire/AutoAugment


from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import logging

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from kornia import image_to_tensor, tensor_to_image
import kornia.augmentation as K
from kornia.geometry.transform import resize

from utils.my_augment import Kornia_Randaugment

logger = logging.getLogger()


class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_tmp = np.array(x)  # HxWxC
        x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = resize(x_out.float() / 255.0, (self.input_size, self.input_size))
        return x_out


class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size=(inp_size, inp_size)),
            K.RandomCrop(size=(inp_size, inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
        )
        # self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        # if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size=(self.inp_size, self.inp_size)),
            K.RandomCrop(size=(self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
        )
        # print("transform")
        # print(self.transforms)
        x_out = self.transforms(x)  # BxCxHxW
        return x_out

def get_transform(dataset, transform_list, method_name, type_name, transform_on_gpu=False, input_size=None):
    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset, type_name=type_name)
    if input_size is not None:
        inp_size = input_size
        
    train_transform = []
    if "cutout" in transform_list:
        train_transform.append(Cutout(size=16))
    if "randaug" in transform_list:
        train_transform.append(transforms.RandAugment())
    if "autoaug" in transform_list:
        if hasattr(transforms, 'AutoAugment'):
            if 'cifar' in dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
        else:
            train_transform.append(select_autoaugment(dataset))
    if "trivaug" in transform_list:
        train_transform.append(transforms.TrivialAugmentWide())
    if transform_on_gpu:
        cpu_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.PILToTensor()
            ])
        if "xder" in method_name:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ConvertImageDtype(torch.float32)
                ]
            )            
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        cpu_transform = None
        if "xder" in method_name:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor()
                ]
            )
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
    logger.info(f"Using train-transforms {train_transform}")

    if "xder" in method_name:
        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return train_transform, test_transform, cpu_transform, n_classes


def select_autoaugment(dataset):
    logger.info(f"{dataset}: autoaugmentation is applied")
    if 'imagenet' in dataset:
        return ImageNetPolicy()
    else:
        return CIFAR10Policy()

class ImageNetPolicy(object):
    """Randomly choose one of the best 24 Sub-policies on ImageNet.
    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.
    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """Randomly choose one of the best 25 Sub-policies on SVHN.
    Example:
    >>> policy = SVHNPolicy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     SVHNPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),
            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class Cutout:
    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[
            upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :
        ] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img


class Invert:
    def __call__(self, pil_img):
        return ImageOps.invert(pil_img)


class Equalize:
    def __call__(self, pil_img):
        return ImageOps.equalize(pil_img)


class Solarize:
    def __init__(self, v):
        assert 0 <= v <= 256
        self.v = v

    def __call__(self, pil_img):
        return ImageOps.solarize(pil_img, self.v)
    
def get_statistics(dataset: str, type_name:str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
        
    assert dataset in [
        "cifar10", 
        "PACS",
        "OfficeHome",
        "DomainNet",
        "birdsnap",
        "aircraft", 
        "food101",
        "cct", 
        "PACS_final"
    ]

    mean = {
        "cifar10": {
                    "c":(0.4914, 0.4822, 0.4465),
                    "nc":(0.52942301, 0.514956, 0.46244773),
                    "ma":(0.48769093, 0.47800379, 0.44362607),
                    "generated":(0.4941789, 0.46011056, 0.41899354),
                    "sdxl_diversified":(0.51821145, 0.47050099, 0.42454597),
                    "web":(0.51270857, 0.50129301, 0.45065171),
                    "web_10":(0.22150516, 0.21842873, 0.22206621),
                    "original_test":(0.49228376, 0.48323634, 0.44864968)
                   },
        "PACS": {
                    "art_painting":(0.55285876, 0.50653457, 0.45607477),
                    "cartoon":(0.80464879, 0.77986176, 0.73312584),
                    "ma":(0.506557, 0.48127243, 0.43791363), #photo
                    "sketch":(0.95286256, 0.95286256, 0.95286256),
                    "generated":(0.49644337, 0.45678366, 0.41565559),
                    "sdxl_diversified":(0.51186689, 0.45341532, 0.40799069),
                    "web":(0.53028615, 0.50012238, 0.44955431),
                    "web_10":(0.50727359, 0.48563996, 0.43021697),
                    "prompt_ensemble":(0.54546987, 0.51522406, 0.47838089)
                    },
        "PACS_final": {
                    "art_painting":(0.55285876, 0.50653457, 0.45607477),
                    "cartoon":(0.80464879, 0.77986176, 0.73312584),
                    "sketch":(0.95286256, 0.95286256, 0.95286256),
                    "train_ma":(0.506557, 0.48127243, 0.43791363),
                    "test_ma":(0.506557, 0.48127243, 0.43791363),
                    "generated":(0.49644337, 0.45678366, 0.41565559),
                    "sdxl_diversified":(0.51186689, 0.45341532, 0.40799069),
                    "web":(0.53028615, 0.50012238, 0.44955431),
                    "web":(0.53028615, 0.50012238, 0.44955431),
                    "web_10":(0.50727359, 0.48563996, 0.43021697),
                    "both_ensembled":(0.51523965, 0.46755071, 0.4216341),
                    "ensembled_RMD_0_5_modelwise":(0.54279138, 0.48778675, 0.43850795),
                    "ensembled_RMD_1_modelwise":(0.5411822, 0.48398797, 0.4330591 ),
                    "ensembled_RMD_3_modelwise":(0.54474629, 0.48422359, 0.43456402),
                    "ensembled_RMD_classwise_temp_0_5":(0.53033692, 0.47375953, 0.42994062),
                    "ensembled_RMD_classwise_temp_1":(0.53485207, 0.47997114, 0.43362797),
                    "ensembled_RMD_classwise_temp_3":(0.53094405, 0.48009067, 0.43317851),
                    },
        "OfficeHome": {
                    "Art":(0.51358405, 0.48082051, 0.44370487),
                    "Clipart":(0.59600234, 0.57472279, 0.54917313),
                    "Product":(0.7443417, 0.73193108, 0.72330176),
                    "train_ma":(0.60439522, 0.57329088, 0.54034423), #Real
                    "test_ma":(0.60439522, 0.57329088, 0.54034423), #Real
                    "generated":(0.54488816, 0.50563005, 0.4752351),
                    "web":(0.59911281, 0.5682033, 0.53703091),
                    },
        "DomainNet": {
                    "infograph":(0.68617464, 0.69427871, 0.66222237),
                    "clipart":(0.73332579, 0.7139194, 0.68080857),
                    "train_ma":(0.61042546, 0.59271156, 0.56087532),
                    "test_ma":(0.61042546, 0.59271156, 0.56087532),
                    "quickdraw":(0.9487493, 0.9487493, 0.9487493), #Real
                    "sketch":(0.82941419, 0.823725, 0.81470561),
                    "painting":(0.57188677, 0.54422885, 0.50600895),
                    "generated":(0.51486457, 0.47512137, 0.43738667),
                    "sdxl_diversified":(0.51486457, 0.47512137, 0.43738667),
                    "web":(0.52883154, 0.5070628, 0.4686003),
                    "web2":(0.58365438, 0.5622869, 0.53059538),
                    },
        "birdsnap": {
                    "ma":(0.48562428, 0.49993579, 0.45427733),
                    "generated":(0.52061567, 0.4907617, 0.42836726),
                    "web":(0.51192757, 0.5201538, 0.47111271),
                    "web_10":(0.51192757, 0.5201538, 0.47111271)
                    },
        "aircraft": {
                    "train_ma":(0.47799557, 0.50874851, 0.53231959),
                    "test_ma":(0.47457604, 0.50624293, 0.5313826),
                    "sdxl_diversified":(0.5243231,  0.50895877, 0.49565841),
                    "web_10":(0.53440468, 0.55515636, 0.57052223),
                    "web":(0.53440468, 0.55515636, 0.57052223)
                    },
        "food101": {
                    "train_ma":(0.5418628, 0.44118891, 0.34277118),
                    "test_ma":(0.5418628, 0.44118891, 0.34277118),
                    "sdxl_diversified":(0.56653057, 0.46730779, 0.38791377),
                    "sdxl_diversified_newprompt":(0.56653057, 0.46730779, 0.38791377),
                    "web":(0.59503791, 0.50928233, 0.41866666),
                    "web_10":(0.59503791, 0.50928233, 0.41866666),
                    },
        "cct": {
                    "train_ma":(0.36634208, 0.36969662, 0.35108814),
                    "in_test_ma":(0.37189878, 0.36819162, 0.33784549),
                    "out_test_ma":(0.35326133, 0.35555819, 0.32220487),
                    "sdxl_diversified":(0.51775118, 0.46570602, 0.41894587),
                    "sdxl_diversified_nofiltering":(0.52440687, 0.47046574, 0.42583795),
                    "generated":(0.48765117, 0.44569423, 0.39488544),
                    "web":(0.52311238, 0.50476414, 0.44740796),
                    "web_10":(0.52311238, 0.50476414, 0.44740796),
                    }
    }
    
    std = {
        "cifar10": {
                    "c":(0.22923981, 0.23307575, 0.26722828),
                    "nc":(0.22417389, 0.21921037, 0.22500719),
                    "ma":(0.20118966, 0.19834845, 0.19964025),
                    "generated":(0.17070625, 0.16452244, 0.16165383),
                    "sdxl_diversified":(0.17194194, 0.16468459, 0.15679837),
                    "web":(0.21963093, 0.21648452, 0.21900275),
                    "web_10":(0.22150516, 0.21842873, 0.22206621),
                    "original_test":(0.20110617, 0.19824355, 0.20024704)
                   },
        "PACS": {
                    "art_painting":(0.22731147, 0.21697391, 0.21904467),
                    "cartoon":(0.23009742, 0.25253542, 0.29657618),
                    "ma":(0.24281495, 0.23492118, 0.24456251), #photo
                    "sketch":(0.18228991, 0.18228991, 0.18228991),
                    "generated":(0.18232649, 0.17583509, 0.1723105),
                    "sdxl_diversified":(0.17562051, 0.16745579, 0.16286352),
                    "web":(0.23207068, 0.23172198, 0.23928677),
                    "web_10":(0.23025856, 0.22669217, 0.23281099),
                    "prompt_ensemble":(0.19371414, 0.18956911, 0.18805951)
                    },
        "PACS_final": {
                    "art_painting":(0.22731147, 0.21697391, 0.21904467),
                    "cartoon":(0.23009742, 0.25253542, 0.29657618),
                    "sketch":(0.18228991, 0.18228991, 0.18228991),
                    "train_ma":(0.24281495, 0.23492118, 0.24456251), #photo
                    "test_ma":(0.24281495, 0.23492118, 0.24456251), #photo
                    "generated":(0.18232649, 0.17583509, 0.1723105),
                    "sdxl_diversified":(0.17562051, 0.16745579, 0.16286352),
                    "web":(0.23207068, 0.23172198, 0.23928677),
                    "web2":(0.23207068, 0.23172198, 0.23928677),
                    "web_10":(0.23025856, 0.22669217, 0.23281099),
                    "both_ensembled":(0.21889653, 0.21000442, 0.20560589),
                    "ensembled_RMD_0_5_modelwise":(0.21534518, 0.20495051, 0.20374349),
                    "ensembled_RMD_1_modelwise":(0.2235222, 0.2101736, 0.20820453),
                    "ensembled_RMD_3_modelwise":(0.21604802, 0.20423867, 0.20078569),
                    "ensembled_RMD_classwise_temp_0_5":(0.2180623, 0.20280125, 0.19698737),
                    "ensembled_RMD_classwise_temp_1":(0.21935218, 0.20703404, 0.2056899),
                    "ensembled_RMD_classwise_temp_3":(0.22964757, 0.2173234, 0.21666681),
                    },
        "OfficeHome": {
                    "Art":(0.23799188, 0.23378391, 0.22740558),
                    "Clipart":(0.29547344, 0.29370141, 0.29833622),
                    "Product":(0.27141979, 0.27763768, 0.28266663),
                    "train_ma":(0.23903598, 0.23973671, 0.24476987),
                    "test_ma":(0.23903598, 0.23973671, 0.24476987),
                    "generated":(0.18006793, 0.1747583, 0.17389996),
                    "web":(0.22672109, 0.22448043, 0.22706106),
                    },
        "DomainNet": {
                    "infograph":(0.24121049, 0.22795033, 0.24102338),
                    "clipart":(0.26243366, 0.26869265, 0.28851051),
                    "train_ma":(0.23142732, 0.23298619, 0.24216565),
                    "test_ma":(0.23142732, 0.23298619, 0.24216565),
                    "quickdraw":(0.20894745, 0.20894745, 0.20894745), #Real
                    "sketch":(0.20092711, 0.20252686, 0.20295023),
                    "painting":(0.22932183, 0.2227554, 0.22761393),
                    "generated":(0.18092263, 0.17377207, 0.17117467),
                    "sdxl_diversified":(0.18092263, 0.17377207, 0.17117467),
                    "web":(0.22322423, 0.22032975, 0.22439602),
                    "web2":(0.22819673, 0.22735438, 0.2337853),
                    },
        "birdsnap": {
                    "ma":(0.16857489, 0.16997163, 0.18407089),
                    "generated":(0.13778179, 0.13694537, 0.142607),
                    "web":(0.17907705, 0.17926208, 0.19245465),
                    "web_10":(0.17907705, 0.17926208, 0.19245465)
                    },
        "aircraft": {
                    "train_ma":(0.18552984, 0.18497887, 0.20751855),
                    "test_ma":(0.18449496, 0.18354592, 0.20612134),
                    "sdxl_diversified":(0.16649881, 0.15924763, 0.15869668),
                    "web":(0.20432787, 0.20202054, 0.21482215),
                    "web_10":(0.20432787, 0.20202054, 0.21482215),
                    },
        "food101": {
                    "train_ma":(0.23137112, 0.24127275, 0.23840478),
                    "test_ma":(0.22816443, 0.23956233, 0.23951101),
                    "sdxl_diversified":(0.20913193, 0.20663952, 0.20379563),
                    "sdxl_diversified_newprompt":(0.20913193, 0.20663952, 0.20379563),
                    "web":(0.22852881, 0.239719, 0.25335559),
                    "web":(0.22852881, 0.239719, 0.25335559),
                    },
        "cct": {
                    "train_ma":(0.12396279, 0.12264526, 0.12559776),
                    "in_test_ma":(0.12986409, 0.12777831, 0.12890333),
                    "out_test_ma":(0.12425986, 0.12297911, 0.12553589),
                    "sdxl_diversified":(0.1735966, 0.16523693, 0.15844997),
                    "sdxl_diversified_nofiltering":(0.17607878, 0.16667338, 0.1596471),
                    "generated":(0.16267202, 0.15566079, 0.15612274),
                    "web":(0.22099038, 0.21976039, 0.22259311),
                    "web_10":(0.22099038, 0.21976039, 0.22259311),
                    }
    }


    classes = {
        "cifar10": 10,
        "PACS": 7,
        "PACS_final": 7,
        "OfficeHome": 65,
        "DomainNet": 345,
        "birdsnap": 500,
        "aircraft": 100,
        "food101": 101,
        "cct": 12
    }

    in_channels = 3
    inp_size = 224
    
    return (
        mean[dataset][type_name],
        std[dataset][type_name],
        classes[dataset],
        inp_size,
        in_channels,
    )