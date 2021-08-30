import os
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import *


def set_gender(gender):
    return 0 if gender == 'male' else 1

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
        df_origin['gender_age_cls'] = df_origin.apply(lambda x : set_age(x['age']) + 3*set_gender(x['gender']) ,axis=1)
        
        train_df, val_df = train_test_split(df_origin, test_size=150, stratify=df_origin.gender_age_cls, random_state=52)

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
        transform 함수를 설정하는 함수입니다.
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
        else : raise Exception(f'파일명 오류 : {file_path}, {files[sub_index]}, {mask}')

        return image, y#, age #age는 사진 수동 검증용, 나이 파악이 힘듬

    def __len__(self):
        return len(self.df_csv)*7


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
