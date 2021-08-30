import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MaskDataset_new_img(Dataset):
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
        file_path = os.path.join(self.main_path, 'new_imgs', sub_path)
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