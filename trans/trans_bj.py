from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap

def basic_train_trans():
    return transforms.Compose([
                            # transforms.CenterCrop(300),
                            transforms.ToTensor(),
                            # transforms.Normalize(0.5, 0.5),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                            ])

def basic_test_trans():
    return transforms.Compose([
                        # transforms.CenterCrop(300),
                        transforms.ToTensor(),
                        # transforms.Normalize(0.5, 0.5),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                        ])

def A_test_trans():
    return transforms.Compose([
                        Ap.ToTensorV2(),
                        A.Normalize(mean=0.5, std=0.2, max_pixel_value=255.0, p=1.0),
                        ])
                        
def A_just_tensor():
    return A.Compose([
                        Ap.ToTensorV2()
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
                        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                        A.GaussNoise(var_limit=(1000, 1600), p=0.2),
                        A.OneOf([
                            A.GridDropout(),
                            A.GlassBlur(),
                            A.GaussianBlur(),
                            A.ColorJitter(),
                            A.Equalize(),
                            A.ChannelDropout(),
                            A.ChannelShuffle(),
                        ]),
                        Ap.ToTensorV2()
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
                        Ap.ToTensorV2()
                    ])

def base_line() :
    return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                    A.GaussNoise(p=0.5),
                    A.Normalize(mean=0.5, std=0.2, max_pixel_value=255.0, p=1.0),
                    Ap.ToTensorV2(p=1.0)
                ])

