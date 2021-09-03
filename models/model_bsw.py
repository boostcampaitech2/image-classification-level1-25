import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class efficientnet_b2_pruned(nn.Module) :
    def __init__(self, num_classes) :
        super().__init__()
        self.model = timm.create_model(model_name= "efficientnet_b2_pruned",
                                        num_classes = num_classes, pretrained = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x                                        

class cspdarknet53(nn.Module) :
    def __init__(self, num_classes) :
        super().__init__()
        self.model = timm.create_model(model_name= "cspdarknet53",
                                        num_classes = 18, pretrained = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x