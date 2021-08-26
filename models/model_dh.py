import timm
import numpy as np
import torch
import torch.nn as nn

class resnetbase2(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.res = timm.create_model(model_name = "resnest101e", num_classes = num_classes, pretrained = True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        return x