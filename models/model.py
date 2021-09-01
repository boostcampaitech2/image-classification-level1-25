import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class resnetbase(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.superM = torchvision.models.resnet18(pretrained=True)
    
        self.superM.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.superM.fc.weight)
        stdv = 1/np.sqrt(512)
        self.superM.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x

class rexnet_200base(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.superM = timm.create_model(model_name = "rexnet_200", # 불러올 모델 architecture,
                                        num_classes=num_classes, # 예측 해야하는 class 수
                                        pretrained = True # 사전학습된 weight 불러오기
                                    )
    
        # self.superM.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # torch.nn.init.xavier_uniform_(self.superM.fc.weight)
        # stdv = 1/np.sqrt(512)
        # self.superM.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x
    
    
class MultiModelMergeModel(nn.Module):
    def __init__(self, modelMASK, modelAGE, modelGENDER,
                    concatclasses : int = 8 , num_classes: int = 18,
                    prev_model_frz=True ):
        super().__init__()
        self.prev_model_frz = prev_model_frz

        self.modelMASK = modelMASK
        self.modelAGE = modelAGE
        self.modelGENDER = modelGENDER
        self.classifier = nn.Sequential(
            nn.Linear(concatclasses, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        
    def forward(self, IMAGE):
        MASK = self.modelMASK(IMAGE)
        AGE = self.modelAGE(IMAGE)
        GENDER = self.modelGENDER(IMAGE)
        if self.prev_model_frz :
            MERGED = torch.cat((MASK.detach(), AGE.detach(), GENDER.detach()), dim=1)
        else :
            MERGED = torch.cat((MASK, AGE, GENDER), dim=1)
        MERGED = self.classifier(nn.functional.relu(MERGED))
        return MERGED


class ensemble(nn.Module):
    def __init__(self, modelname: str, length: int, device, num_classes: int = 1000):
        super().__init__()
        self.superM = []
        for _ in range(length):
            self.superM.append(globals()[modelname](num_classes=num_classes).to(device))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.nn.Softmax(dim=1)
        result = []
        for idx, M in enumerate(self.superM):
            result.append(s(M(x))) 
        result = torch.stack(result, dim=0)
        return torch.sum(result, dim=0)
    
