import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import *
import albumentations as A
import albumentations.pytorch as Ap

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


class BaseAugmentation:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

 
class CustomAugmentation:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

        
class A_resize_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.Normalize(mean=self.mean, std=self.std),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)

class My_trans1 :
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
            A.CenterCrop(height=384, width=384),
            A.Resize(width=resize[0], height=resize[1]),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(),
            A.Normalize(mean=self.mean, std=self.std),
            Ap.ToTensorV2(),
        ])
    def __call__(self, image):
        return self.transform(image=image)

class My_trans2 :
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
            A.CenterCrop(height=384, width=384),
            A.Resize(width=resize[0], height=resize[1]),
            A.ShiftScaleRotate(p=0.7),
            A.GaussNoise(var_limit=(400, 600), p=0.1),
            A.Normalize(mean=self.mean, std=self.std),
            Ap.ToTensorV2(),
        ])
    def __call__(self, image):
        return self.transform(image=image)

class My_trans3 :
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
            A.CenterCrop(height=384, width=384),
            A.Resize(width=resize[0], height=resize[1]),
            A.ShiftScaleRotate(p=0.7),
            A.GaussianBlur(),
            A.Normalize(mean=self.mean, std=self.std),
            Ap.ToTensorV2(),
        ])
    def __call__(self, image):
        return self.transform(image=image)

class My_trans4 :
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
            A.CenterCrop(height=300, width=250),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=8, max_width=8,
                min_holes=None, min_height=None, min_width=None,
                fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
            A.Resize(width=resize[0], height=resize[1]),
            A.Normalize(mean=self.mean, std=self.std),
            A.HorizontalFlip(p=0.5),
            Ap.ToTensorV2(),
        ])
    def __call__(self, image):
        return self.transform(image=image)


class My_trans_valid:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.Normalize(mean=self.mean, std=self.std),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)


class A_centercrop_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.Normalize(mean=self.mean, std=self.std),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)


class A_simple_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.RandomCrop(width=300, height=300, p=0.5),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.7),
                                    A.Normalize(mean=self.mean, std=self.std),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)



class A_random_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.RandomCrop(height=256, width=256, p=0.5),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.7),
                                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                                    A.GaussNoise(var_limit=(400, 600), p=0.1),
                                    A.OneOf([
                                        A.GridDropout(),
                                        A.GlassBlur(),
                                        A.GaussianBlur(),
                                        A.ColorJitter(),
                                        A.Equalize(),
                                        A.ChannelDropout(),
                                        A.ChannelShuffle(),
                                    ], p=0.2),
                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)

class A_random_custom_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.7),
                                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                                    A.GaussNoise(var_limit=(400, 600), p=0.1),
                                    A.OneOf([
                                        A.GridDropout(),
                                        A.GlassBlur(),
                                        A.GaussianBlur(),
                                        A.ColorJitter(),
                                        A.Equalize(),
                                        A.ChannelDropout(),
                                        A.ChannelShuffle(),
                                    ], p=0.2),
                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)


class A_random_trans_no_cut():
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(p=0.7),
                            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                            A.GaussNoise(var_limit=(400, 600), p=0.1),
                            A.OneOf([
                                A.GridDropout(),
                                A.GlassBlur(),
                                A.GaussianBlur(),
                                A.ColorJitter(),
                                A.Equalize(),
                                A.ChannelDropout(),
                                A.ChannelShuffle(),
                            ], p=0.2),
                            A.Normalize(),
                            Ap.ToTensorV2(),
                        ])

class A_just_tensor():
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                            A.Normalize(),
                            Ap.ToTensorV2(),
                        ])
