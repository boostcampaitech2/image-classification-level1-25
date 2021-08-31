import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import *
import albumentations as A
import albumentations.pytorch as Ap
import torchvision.transforms.functional as F

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

class A_trans_train:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.RandomCrop(width=300, height=300, p=0.5),
                                    A.Resize(width=resize[0], height=resize[1]),
                                    A.Normalize(mean=self.mean, std=self.std),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)

class A_trans_val:
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

class trans_crop:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        image = F.crop(img=transforms.ToPILImage()(image),top=50,left=80,height=320,width=220)
        return self.transform(image)