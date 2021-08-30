import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm


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


if __name__ == '__main__':
    M = rexnet_200base(num_classes = 18)
    print(M)