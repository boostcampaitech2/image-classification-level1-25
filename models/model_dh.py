from typing import List
import timm
import numpy as np
import torch
import torch.nn as nn
import torchvision


class resnet101e(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_model = timm.create_model(model_name = "resnest101e", num_classes = 18, pretrained = True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_model(x)
        return x

class resnet101e(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.superM = timm.create_model(model_name = "resnest101e", pretrained = True)
    
        self.superM.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.superM.fc.weight)
        stdv = 1/np.sqrt(2048)
        self.superM.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x
    

class rexnet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.superM = timm.create_model(model_name = "rexnet_200", pretrained = True)
    
        # self.superM.fc = torch.nn.Linear(in_features=2560, out_features=num_classes, bias=True)
        # torch.nn.init.xavier_uniform_(self.superM.fc.weight)
        # stdv = 1/np.sqrt(2560)
        # self.superM.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
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


