import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init

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

        initialize_weights(self.superM.classifier)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.superM(x)
        return x

def initialize_weights(model):
        """
        Xavier uniform 분포로 모든 weight 를 초기화합니다.
        더 많은 weight 초기화 방법은 다음 문서에서 참고해주세요. https://pytorch.org/docs/stable/nn.init.html
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


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
    
 