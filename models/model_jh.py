import numpy as np
import torch
import torch.nn as nn
import torchvision

class vggnetbase(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.superM = torchvision.models.vgg16(pretrained=True)
    
        self.superM.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)
        torch.nn.init.xavier_uniform_(self.superM.fc.weight)
        stdv = 1/np.sqrt(512)
        self.superM.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x
