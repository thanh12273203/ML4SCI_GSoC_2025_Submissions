# Import necessary dependencies
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

# ResNet15 architecture
class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet15(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        inplanes: int = 64,
        planes: int = 64,
        layers: List[int] = [3, 1, 3]
    ):
        super(ResNet15, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = self._make_layer(planes, layers[0], stride=1)
        self.layer2 = self._make_layer(planes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(planes*4, layers[2], stride=2)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(planes*4, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
# Parallel ResNet15 architecture
class ParallelResNet15(nn.Module):
    def __init__(self, num_classes: int = 2, inplanes: int = 64, planes: int = 64, layers=[3, 1, 3]) -> None:
        super(ParallelResNet15, self).__init__()
        self.resnet_energy = ResNet15(num_classes=planes*4, in_channels=1, inplanes=inplanes, planes=planes, layers=layers)
        self.resnet_time = ResNet15(num_classes=planes*4, in_channels=1, inplanes=inplanes, planes=planes, layers=layers)
        self.fc = nn.Linear(self.resnet_energy.fc.out_features*2, num_classes)

    def forward(self, x_energy: Tensor, x_time: Tensor) -> Tensor:
        x_energy = self.resnet_energy(x_energy)
        x_time = self.resnet_time(x_time)
        out = torch.cat((x_energy, x_time), dim=1)
        out = self.fc(out)

        return out
    
# Expander architecture for VIC REgularization
class Expander(nn.Module):
    def __init__(self, representation_dim: int = 512, hidden_dim: int = 2048):
        super(Expander, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(representation_dim)
        self.linear1 = nn.Linear(representation_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.embedding_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.batch_norm1(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.batch_norm2(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.embedding_layer(x)

        return x
    
# Architecture for the VICReg training pipeline
class ResNet15WithExpander(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        inplanes: int = 64,
        planes: int =64,
        layers: List[int] = [2, 2, 3],
        representation_dim: int = 512,
        hidden_dim: int = 2048
    ):
        super(ResNet15WithExpander, self).__init__()
        self.resnet = ResNet15(num_classes=representation_dim, in_channels=in_channels, inplanes=inplanes, planes=planes, layers=layers)
        self.expander = Expander(representation_dim=representation_dim, hidden_dim=hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.resnet(x)
        x = self.expander(x)

        return x