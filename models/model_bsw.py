import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class vit_large_patch16_384(nn.Module) :
    def __init__(self, n_class, pretrained=False) :
        super().__init()
        self.model = timm.create_model(model_name= "vit_large_patch16_384",
                                        num_classes = n_class, pretrained = pretrained)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x                                        

class vit_small_patch16_224(nn.Module) :
    def __init__(self, num_classes) :
        super().__init()
        self.model = timm.create_model(model_name= "vit_small_patch16_224",
                                        num_classes = 18, pretrained = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class cspdarknet53(nn.Module) :
    def __init__(self, num_classes) :
        super().__init()
        self.model = timm.create_model(model_name= "cspdarknet53",
                                        num_classes = 18, pretrained = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x