import numpy as np
import pandas as pd
import os
from glob import glob 
from PIL import Image
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split


import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

class GenderLabels:
    male = 0
    female = 1

class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1.jpg": MaskLabels.mask,
        "mask2.jpg": MaskLabels.mask,
        "mask3.jpg": MaskLabels.mask,
        "mask4.jpg": MaskLabels.mask,
        "mask5.jpg": MaskLabels.mask,
        "incorrect_mask.jpg": MaskLabels.incorrect,
        "normal.jpg": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, img_dir, transform=None):
        """
        MaskBaseDataset을 initialize 합니다.
        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            transform: Augmentation을 하는 함수입니다.
        """
        self.img_dir = img_dir
        #TO-DO
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.2, 0.2, 0.2)
        self.transform = transform

        self.setup()

    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
        
    def setup(self):
        """
        image의 경로와 각 이미지들의 label을 계산하여 저장해두는 함수입니다.
        """
        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러옵니다.
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        # 레이블을 불러옵니다.
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)



# class MaskLabels:
#     mask = 0
#     incorrect = 1
#     normal = 2

# class GenderLabels:
#     male = 0
#     female = 1

# class AgeGroup:
#     map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2


# class Dataset:
#     num_classes = 3 * 2 * 3

#     _file_names = {
#         "mask1.jpg": MaskLabels.mask,
#         "mask2.jpg": MaskLabels.mask,
#         "mask3.jpg": MaskLabels.mask,
#         "mask4.jpg": MaskLabels.mask,
#         "mask5.jpg": MaskLabels.mask,
#         "incorrect_mask.jpg": MaskLabels.incorrect,
#         "normal.jpg": MaskLabels.normal
#     }

#     image_paths = []
#     mask_labels = []
#     gender_labels = []
#     age_labels = []

#     def __init__(self, path = None,
#     test_size = 0.3,
#     train = True,
#     transform = None,
#     shuffle = False,
#     random_state = None, 
#     stratify = None) :

#         # Seed 설정
#         random.seed(random_state)
#         np.random.seed(random_state)

#         # Data 
#         train_df = pd.read_csv(path)
#         if train :
#             self.df = train_test_split(train_df, test_size = test_size,
#                                         random_state = random_state,
#                                         shuffle = shuffle,
#                                         stratify = stratify)

#         self.transform = transform
#         # Labeling 
#         self.setup()



#         train_df = pd.read_csv(path)

    

#     def setup(self) :
#         profiles = os.listdir(self.img_dir)
#         for profile in profiles:
#             for file_name, label in self._file_names.items():
#                 img_path = os.path.join(self.img_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
#                 if os.path.exists(img_path):
#                     self.image_paths.append(img_path)
#                     self.mask_labels.append(label)

#                     id, gender, race, age = profile.split("_")
#                     gender_label = getattr(GenderLabels, gender)
#                     age_label = AgeGroup.map_label(age)

#                     self.gender_labels.append(gender_label)
#                     self.age_labels.append(age_label)




