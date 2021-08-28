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

        train_share = 0.9
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


class basicDatasetA(Dataset):
    def __init__(self, data_path, train = 'ALL', transform = None):
        self.main_path = data_path
        self.transform = transform
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        df_origin['gender'] = df_origin['gender'].map({'male':0, 'female':1})
        df_origin = df_origin.sample(frac=1).reset_index(drop=True)

        train_share = 0.9
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


def set_gender(gender):
    return 0 if gender == 'male' else 1

def set_age(age):
    if age<30:
        return 0
    elif age<60:
        return 1
    else:
        return 2

class EvalTestDatasetA(Dataset): #미완성
    def __init__(self, data_path, train = 'ALL', transform = None, N = 0):
        self.main_path = data_path
        self.transform = transform
        df_origin = pd.read_csv(os.path.join(self.main_path, 'train.csv'))
        print(df_origin['age'][0])
        print(set_age(df_origin['age'][0]))
        print('-'*50)
        df_origin['gender_age_cls'] = df_origin.apply(lambda x : set_age(x['age']) + 3*set_gender(x['gender']) ,axis=1)

        from sklearn.model_selection import train_test_splitx
        train_df, val_df = train_test_split(df_origin, test_size=150, stratify=df_origin.gender_age_cls, random_state=52)

        train_df, fold1 = train_test_split(df_origin, test_size=20%, stratify=df_origin.gender_age_cls, random_state=52)
        train_df, fold2 = train_test_split(train_df, test_size=25%, stratify=df_origin.gender_age_cls, random_state=52)
        train_df, fold3 = train_test_split(train_df, test_size=33%, stratify=df_origin.gender_age_cls, random_state=52)
        fold5, fold4 = train_test_split(train_df, test_size=50%, stratify=df_origin.gender_age_cls, random_state=52)

        from sklearn.model_selection import StratifiedKFold
        train_ids, val_ids = [], []
        for x, y in StratifiedKFold().split():
            train_ids.append(x)
            val_ids.append(y)




        total = len(df_origin)
        if train == 'ALL':
            self.df_csv = df_origin
        elif train == 'train' :
            self.df_csv = train_df
        elif train == 'test' :
            self.df_csv = val_df

        elif train == 'fold' : 
            if state == 'train':
                self.df_csv = train_df[train_ids[fold_idx]]
            elif state == 'vaild':
                self.df_csv = train_df[val_ids[fold_idx]]

        else :
            raise Exception(f'train error {train} not in [''ALL'', ''train'', ''test'']')
        




    def __getitem__(self, index):
        # return self.df_csv.iloc[index]
        return self.df_csv

    def __len__(self):
        return len(self.df_csv)*7