
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap

def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            CenterCrop(img_size[1],img_size[1]),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = Compose([
            CenterCrop(img_size[1],img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations

def A_resize_trans():
    return A.Compose([
                        A.CenterCrop(height=384, width=384),
                        A.Resize(height=256, width=256),
                        A.Normalize(),
                        Ap.ToTensorV2(),
                    ])