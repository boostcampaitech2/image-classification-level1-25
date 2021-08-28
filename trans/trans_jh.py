from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap

def resize_244_random__trans():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
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