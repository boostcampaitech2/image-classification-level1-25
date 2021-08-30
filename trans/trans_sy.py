from albumentations import *
from albumentations.pytorch import ToTensorV2
import albumentations as A
import albumentations.pytorch as Ap
from albumentations.augmentations.transforms import Normalize
from torchvision import transforms

def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
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
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
        
def basic_train_trans(img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    return Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def basic_test_trans(img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    return Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)



def A_just_tensor():
    return A.Compose([
                        A.Normalize(),
                        Ap.ToTensorV2(),
                    ])


def A_resize_trans():
    return A.Compose([
                        A.CenterCrop(height=384, width=384),
                        A.Resize(height=256, width=256),
                        A.Normalize(),
                        Ap.ToTensorV2(),
                    ])


def A_centercrop_trans():
    return A.Compose([
                        A.CenterCrop(height=300, width=300),
                        A.Resize(height=256, width=256),
                        A.Normalize(),
                        Ap.ToTensorV2(),
                    ])


def A_random_trans():
    return A.Compose([
                        A.CenterCrop(height=384, width=384),
                        A.OneOf([
                            A.RandomCrop(height=256, width=256),
                            A.Resize(height=256, width=256),
                        ], p=1.0),
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

def A_simple_trans():
    return A.Compose([
                        A.CenterCrop(height=384, width=384),
                        A.OneOf([
                            A.RandomCrop(height=256, width=256),
                            A.Resize(height=256, width=256),
                        ], p=1.0),
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(p=0.7),
                        A.Normalize(),
                        Ap.ToTensorV2(),
                    ])


def A_random_trans_no_cut():
    return A.Compose([
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


def A_random_trans_cut():
    return A.Compose([
                        A.CenterCrop(height=384, width=384),
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