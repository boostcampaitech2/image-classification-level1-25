import torchvision
from torchvision import transforms

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
