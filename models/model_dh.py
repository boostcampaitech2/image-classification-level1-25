import timm
import numpy as np
import torch
import torch.nn as nn



class resnet101e(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_model = timm.create_model(model_name = "resnest101e", num_classes = 18, pretrained = True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_model(x)
        return x

