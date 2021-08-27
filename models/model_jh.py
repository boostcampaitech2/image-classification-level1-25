import numpy as np
import torch
import torch.nn as nn
import torchvision

class vggnetbase(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.superM = torchvision.models.vgg19(pretrained=True)
    
        self.superM.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x
