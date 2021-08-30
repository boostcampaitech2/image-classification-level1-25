import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm

# my model

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
    
 
class resnetbase2(nn.Module):
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

class resnetbase3(nn.Module) :
    def __init__(self, numclasses: int = 1000) :
        super().__init__()
        self.superM = torchvision.models.resnet18(pretrained=True)
        self.linear_layers = nn.Linear(1000,numclasses)
    
    def  forward(self,x) :
        x = self.superM(x)
        return self.linear_layers(x)

class rexnet_200base(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.superM = timm.create_model(model_name = "rexnet_200", # 불러올 모델 architecture,
                                        num_classes=num_classes, # 예측 해야하는 class 수
                                        pretrained = True # 사전학습된 weight 불러오기
                                    )

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

class MergeFreezeModel(nn.Module):
    def __init__(self, modelMASK, modelAGE, modelGENDER,
                    concatclasses : int = 8 , numclasses: int = 18):
        super().__init__()
        self.modelMASK = modelMASK
        self.modelAGE = modelAGE
        self.modelGENDER = modelGENDER
        self.classifier = nn.Sequential(
            nn.Linear(concatclasses, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, numclasses),
        )
        
    def forward(self, IMAGE):
        MASK = self.modelMASK(IMAGE)
        AGE = self.modelAGE(IMAGE)
        GENDER = self.modelGENDER(IMAGE)
        MERGED = torch.cat((MASK.detach(), AGE.detach(), GENDER.detach()), dim=1)
        MERGED = self.classifier(nn.functional.relu(MERGED))
        return MERGED

class MergeFullModel(nn.Module):
    def __init__(self, modelMASK, modelAGE, modelGENDER,
                    concatclasses : int = 8 , numclasses: int = 18):
        super().__init__()
        self.modelMASK = modelMASK
        self.modelAGE = modelAGE
        self.modelGENDER = modelGENDER
        self.classifier = nn.Sequential(
            nn.Linear(concatclasses, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, numclasses),
        )
        
    def forward(self, IMAGE):
        MASK = self.modelMASK(IMAGE)
        AGE = self.modelAGE(IMAGE)
        GENDER = self.modelGENDER(IMAGE)
        MERGED = torch.cat((MASK, AGE, GENDER), dim=1)
        MERGED = self.classifier(nn.functional.relu(MERGED))
        return MERGED