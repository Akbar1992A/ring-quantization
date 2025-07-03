"""
Standard FP32 Models

This module contains the standard, full-precision (FP32) model architectures
for creating baseline results. These models use standard PyTorch layers.

- BasicBlockFP32: Standard building block for ResNet
- ResNetFP32: Base FP32 ResNet architecture
- ResNet20_FP32, ResNet32_FP32, ResNet44_FP32: Specific configurations for CIFAR-10
"""

import torch.nn as nn
import torch.nn.functional as F

class BasicBlockFP32(nn.Module):
    """Standard FP32 Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Use standard nn.Conv2d layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFP32(nn.Module):
    """Standard FP32 ResNet architecture for CIFAR-10"""
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 16
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three groups of residual blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Final classifier
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a layer with multiple blocks"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize BatchNorm and Linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20_FP32(num_classes=10):
    """Standard FP32 ResNet-20 for CIFAR-10"""
    return ResNetFP32(BasicBlockFP32, [3, 3, 3], num_classes=num_classes)


def ResNet32_FP32(num_classes=10):
    """Standard FP32 ResNet-32 for CIFAR-10"""
    return ResNetFP32(BasicBlockFP32, [5, 5, 5], num_classes=num_classes)


def ResNet44_FP32(num_classes=10):
    """Standard FP32 ResNet-44 for CIFAR-10"""
    return ResNetFP32(BasicBlockFP32, [7, 7, 7], num_classes=num_classes)