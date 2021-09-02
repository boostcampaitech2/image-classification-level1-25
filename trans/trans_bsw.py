from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.transforms import HorizontalFlip
from cv2 import GaussianBlur
import torch
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap

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
        


# python train.py --epochs 20 --usertrans trans_bsw --trainaug A_custom_trans --validaug A_trans_val --usermodel model_bsw --model efficientnet_b2_pruned --mode ALL --name bsw/effi_gaussNoise
class A_custom_trans_RandomBrightnessContrast:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    # A.GaussNoise(),
                                    # A.GaussianBlur(),
                                    # A.HorizontalFlip(),

                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)

class A_custom_trans_GaussNoise:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    A.GaussNoise(),
                                    # A.GaussianBlur(),
                                    # A.HorizontalFlip(),

                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)
        

class A_custom_trans_GaussianBlur:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    # A.GaussNoise(),
                                    A.GaussianBlur(),
                                    # A.HorizontalFlip(),

                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)
        

class A_custom_trans_HorizontalFlip:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    # A.GaussNoise(),
                                    # A.GaussianBlur(),
                                    A.HorizontalFlip(),

                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)
        

class A_custom_trans_ShiftScaleRotate:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    # A.GaussNoise(),
                                    # A.GaussianBlur(),
                                    # A.HorizontalFlip(),
                                    A.ShiftScaleRotate(),

                                    A.Normalize(),
                                    Ap.ToTensorV2(),
                                ])

    def __call__(self, image):
        return self.transform(image=image)
        

class A_custom_trans:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.mean = mean
        self.std = std
        self.transform = A.Compose([
                                    A.CenterCrop(height=384, width=384),
                                    A.Resize(width=224, height=224),

                                    # 추가되는 부분
                                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                    # A.GaussNoise(),
                                    # A.GaussianBlur(),
                                    # A.HorizontalFlip(),

                                    A.Normalize(),
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
