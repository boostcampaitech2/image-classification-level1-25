import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from sklearn.model_selection import train_test_split
from trans.trans import A_centercrop_trans

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def set_gender(gender):
    return 0 if gender == 'male' else 1

def set_age(age):
    if age<30:
        return 0
    elif age<60:
        return 1
    else:
        return 2

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mode='ALL', transform = None, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = transform
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." ??? ???????????? ????????? ???????????????
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform ???????????? ???????????? transform ??? ??????????????????"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label
        
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label
    
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp
    
    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        ??????????????? train ??? val ??? ????????????,
        pytorch ????????? torch.utils.data.random_split ????????? ????????????
        torch.utils.data.Subset ????????? ?????? ????????????.
        ????????? ????????? ????????? ????????? ?????? IDE (e.g. pycharm) ??? navigation ????????? ?????? ????????? ??? ??? ???????????? ?????? ??????????????????^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val ????????? ????????? ???????????? ????????? random ??? ??????
        ??????(profile)??? ???????????? ????????????.
        ????????? val_ratio ??? ?????? train / val ????????? ?????? ????????? ????????? ?????? ??????(profile)??? ????????? ???????????? indexing ??? ?????????
        ?????? `split_dataset` ?????? index ??? ?????? Subset ?????? dataset ??? ???????????????.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), transfrom = None):
        self.img_paths = img_paths
        if transfrom:
            self.transform = transfrom
        else:
            self.transform = transforms.Compose([
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class TestDatasetA(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), transform = None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image'].float()
        return image

    def __len__(self):
        return len(self.img_paths)


class basicDatasetA(Dataset):
    num_classes = 3 * 2 * 3
    
    def __init__(self, data_dir, mode = 'ALL', transform = None):
        self.main_path = data_dir
        self.transform = transform
        self.mode = mode
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin['gender_age_cls'] = df_origin.apply(lambda x : set_age(x['age']) + 3*x['gender'] ,axis=1)

        train_df, eval_df = train_test_split(df_origin, test_size=150, stratify=df_origin.gender_age_cls, random_state=25)
        train_df.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)

        if mode == 'ALL':
            self.df_csv = df_origin
        elif mode == 'train' :
            self.df_csv = train_df
        elif mode == 'eval' :
            self.df_csv = eval_df
        else :
            raise Exception(f'train error {mode} not in [ALL, train, eval]')
        
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        main_index, sub_index = index//7, index%7
        sub_path = self.df_csv.iloc[main_index]['path']
        file_path = os.path.join(self.main_path, 'images', sub_path)
        files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']

        image = Image.open(os.path.join(file_path, files[sub_index]))

        if self.transform:
            image = self.transform(image=np.array(image))['image'].float()

        y = 0

        if (age := self.df_csv.iloc[main_index]['age']) < 30: pass
        elif age >= 30 and age < 60: y += 1
        else : y += 2

        y += self.df_csv.iloc[main_index]['gender'] * 3

        if (mask := files[sub_index][0]) == 'm' : pass
        elif mask == 'i' : y += 6
        elif mask == 'n' : y += 12
        else : raise Exception(f'????????? ?????? : {file_path}, {files[sub_index]}, {mask}')

        return image, y#, age #age??? ?????? ?????? ?????????, ?????? ????????? ??????

    def __len__(self):
        return len(self.df_csv)*7


# double data
class basicDatasetA2(Dataset):
    num_classes = 3 * 2 * 3
    
    def __init__(self, data_dir, mode = 'ALL', transform = None):
        self.main_path = data_dir
        self.origin_trans = A_centercrop_trans()
        self.transform = transform
        self.mode = mode
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin['gender_age_cls'] = df_origin.apply(lambda x : set_age(x['age']) + 3*x['gender'] ,axis=1)

        train_df, eval_df = train_test_split(df_origin, test_size=150, stratify=df_origin.gender_age_cls, random_state=25)
        train_df.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)

        if mode == 'ALL':
            self.df_csv = df_origin
        elif mode == 'train' :
            self.df_csv = train_df
        elif mode == 'eval' :
            self.df_csv = eval_df
        else :
            raise Exception(f'train error {mode} not in [ALL, train, eval]')
        
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        if index >= len(self.df_csv)*7:
            is_trans = True
            index = index - len(self.df_csv)*7
        else :
            is_trans = False
        main_index, sub_index = index//7, index%7
 
        sub_path = self.df_csv.iloc[main_index]['path']
        file_path = os.path.join(self.main_path, 'images', sub_path)
        files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']

        image = Image.open(os.path.join(file_path, files[sub_index]))
        if is_trans:
            image = self.origin_trans(image=np.array(image))['image'].float()
        elif not is_trans and self.transform:
            image = self.transform(image=np.array(image))['image'].float()

        

        y = 0

        if (age := self.df_csv.iloc[main_index]['age']) < 30: pass
        elif age >= 30 and age < 60: y += 1
        else : y += 2

        y += self.df_csv.iloc[main_index]['gender'] * 3

        if (mask := files[sub_index][0]) == 'm' : pass
        elif mask == 'i' : y += 6
        elif mask == 'n' : y += 12
        else : raise Exception(f'????????? ?????? : {file_path}, {files[sub_index]}, {mask}')

        return image, y#, age #age??? ?????? ?????? ?????????, ?????? ????????? ??????

    def __len__(self):
        return len(self.df_csv)*7*2

# def set_gender(gender):
#     return 0 if gender == 'male' else 1

def set_age(age):
    if age<30:
        return 0
    elif age<60:
        return 1
    else:
        return 2

class teamDataset(Dataset):
    num_classes = 3 * 2 * 3

    def __init__(self, data_path, train = 'train',mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.main_path = data_path
        self.mean = mean
        self.std = std

        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin['gender_age_cls'] = df_origin.apply(lambda x : set_age(x['age']) + 3*x['gender'] ,axis=1)
        
        train_df, val_df = train_test_split(df_origin, test_size=150, stratify=df_origin.gender_age_cls, random_state=25)

        if train == 'train':
            self.df_csv = train_df
        elif train == 'eval':
            self.df_csv = val_df
        elif train == 'ALL':
            self.df_csv = df_origin
        else:
            raise Exception(f'train error {train} not in [''ALL'', ''train'', ''test'']')
    
    def set_transform(self, transform):
        """
        transform ????????? ???????????? ???????????????.
        """
        self.transform = transform

    def __getitem__(self, index):
        main_index, sub_index = index//7, index%7
        sub_path = self.df_csv.iloc[main_index]['path']
        file_path = os.path.join(self.main_path, 'images', sub_path)
        files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']

        image = Image.open(os.path.join(file_path, files[sub_index]))

        if self.transform:
            image = self.transform(image=np.array(image))['image'].float()

        y = 0

        if (age := self.df_csv.iloc[main_index]['age']) < 30: pass
        elif age >= 30 and age < 60: y += 1
        else : y += 2

        y += self.df_csv.iloc[main_index]['gender'] * 3

        if (mask := files[sub_index][0]) == 'm' : pass
        elif mask == 'i' : y += 6
        elif mask == 'n' : y += 12
        else : raise Exception(f'????????? ?????? : {file_path}, {files[sub_index]}, {mask}')

        return image, y#, age #age??? ?????? ?????? ?????????, ?????? ????????? ??????

    def __len__(self):
        return len(self.df_csv)*7


class MMteamDataset(Dataset):
    num_classes = {'mask' : 3, 'gender' : 2, 'age' : 3, 'merged' : 3*2*3, 'concat': 3+2+3}
    def __init__(self, data_dir, mode = 'ALL', transform = None, val_ratio = 0.1):
        self.main_path = data_dir
        self.transform = transform
        self.mode = mode
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin = df_origin.sample(frac=1).reset_index(drop=True)

        train_share = 1 - val_ratio
        total = len(df_origin)
        if mode == 'ALL':
            self.df_csv = df_origin
        elif mode == 'train' :
            self.df_csv = df_origin.head(int(total*train_share)) 
        elif mode == 'valid' :
            self.df_csv = df_origin.tail(total-int(total*train_share))
        else :
            raise Exception(f'train error {mode} not in [''ALL'', ''train'', ''test'']')
    
    def set_transform(self, transform):
        """
        transform ????????? ???????????? ???????????????.
        """
        self.transform = transform

    def __getitem__(self, index):
        main_index, sub_index = index//7, index%7
        sub_path = self.df_csv.iloc[main_index]['path']
        file_path = os.path.join(self.main_path, 'images', sub_path)
        files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']

        image = Image.open(os.path.join(file_path, files[sub_index]))

        if self.transform:
            image = self.transform(image=np.array(image))['image'].float()
        
        labed_dict = {}
        y = 0

        if (age := self.df_csv.iloc[main_index]['age']) < 30:
            labed_dict['age'] = 0
        elif age >= 30 and age < 60:
            y += 1
            labed_dict['age'] = 1
        else :
            y += 2
            labed_dict['age'] = 2

        y += self.df_csv.iloc[main_index]['gender'] * 3
        labed_dict['gender'] = self.df_csv.iloc[main_index]['gender']

        if (mask := files[sub_index][0]) == 'm' :
            labed_dict['mask'] = 0
        elif mask == 'i' :
            y += 6
            labed_dict['mask'] = 1
        elif mask == 'n' :
            y += 12
            labed_dict['mask'] = 2
        else : raise Exception(f'????????? ?????? : {file_path}, {files[sub_index]}, {mask}')
        
        labed_dict['merged'] = y
        return image, labed_dict

    def __len__(self):
        return len(self.df_csv)*7