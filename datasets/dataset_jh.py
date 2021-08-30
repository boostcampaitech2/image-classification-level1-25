import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MaskDataset(Dataset):
    def __init__(self, data_path, train = 'ALL', transform = None):
        self.main_path = data_path
        self.transform = transform
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})

        total = len(df_origin)
        if train == 'ALL':
            self.df_csv = df_origin
        elif train == 'train' :
            self.df_csv = df_origin.head(int(total*0.8)) 
        elif train == 'test' :
            self.df_csv = df_origin.tail(total-int(total*0.8))
        else :
            raise Exception(f'train error {train} not in [''ALL'', ''train'', ''test'']')
        

    def __getitem__(self, index):
        main_index, sub_index = index//7, index%7
        sub_path = self.df_csv.iloc[main_index]['path']
        file_path = os.path.join(self.main_path, 'images', sub_path)
        files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']

        image = Image.open(os.path.join(file_path, files[sub_index]))

        if self.transform:
            image = self.transform(image)

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
    
  
class MaskDatasetA(Dataset):
    def __init__(self, data_path, train = 'ALL', transform = None):
        self.main_path = data_path
        self.transform = transform
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})

        total = len(df_origin)
        if train == 'ALL':
            self.df_csv = df_origin
        elif train == 'train' :
            self.df_csv = df_origin.head(int(total*0.8)) 
        elif train == 'test' :
            self.df_csv = df_origin.tail(total-int(total*0.8))
        else :
            raise Exception(f'train error {train} not in [''ALL'', ''train'', ''test'']')
        

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



class basicDatasetA(Dataset):
    def __init__(self, data_path, train = 'ALL', transform = None):
        self.main_path = data_path
        self.transform = transform
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin = df_origin.sample(frac=1).reset_index(drop=True)

        train_share = 0.8
        total = len(df_origin)
        if train == 'ALL':
            self.df_csv = df_origin
        elif train == 'train' :
            self.df_csv = df_origin.head(int(total*train_share)) 
        elif train == 'test' :
            self.df_csv = df_origin.tail(total-int(total*train_share))
        else :
            raise Exception(f'train error {train} not in [''ALL'', ''train'', ''test'']')
        

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