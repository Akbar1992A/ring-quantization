"""
Ring Quantization Model Architectures

This module contains the model architectures that use the custom
ring quantization layers defined in `layers.py`.

- BasicBlock: The standard residual building block for ResNet, adapted for ring quantization.
- RingResNet: The base ResNet architecture for CIFAR-10, using ring layers.
- ResNet20, ResNet32, ResNet44: Factory functions to create specific ResNet configurations.
"""

import torch.nn as nn
import torch.nn.functional as F
from .layers import RingConv2d

class BasicBlock(nn.Module):
    """
    A Basic Residual Block for ResNet, using RingConv2d layers.
    This block is the fundamental component of the ResNet architecture.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, ring_size=8, ring_type='triangle'):
        super().__init__()
        
        # The first convolutional layer of the block
        self.conv1 = RingConv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                               padding=1, ring_size=ring_size, ring_type=ring_type, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # The second convolutional layer of the block
        self.conv2 = RingConv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                               padding=1, ring_size=ring_size, ring_type=ring_type, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to allow gradients to flow and to learn residual mappings.
        # It adapts the input dimensions if they change (e.g., due to stride).
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                RingConv2d(in_channels, self.expansion * out_channels, kernel_size=1, 
                          stride=stride, padding=0, ring_size=ring_size, ring_type=ring_type, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # Add shortcut connection
        out += self.shortcut(x)
        # Final activation
        out = F.relu(out, inplace=True)
        return out


class RingResNet(nn.Module):
    """
    A ResNet architecture for CIFAR-10 that uses Ring Quantization layers throughout.
    The structure follows the original ResNet paper's recommendations for CIFAR-10.
    """
    def __init__(self, block, num_blocks, num_classes=10, ring_size=8, ring_type='triangle'):
        super().__init__()
        self.in_channels = 16
        
        # The initial convolutional layer processes the input image
        self.conv1 = RingConv2d(3, 16, kernel_size=3, stride=1, padding=1, 
                               ring_size=ring_size, ring_type=ring_type, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Stack of residual blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, ring_size=ring_size, ring_type=ring_type)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, ring_size=ring_size, ring_type=ring_type)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, ring_size=ring_size, ring_type=ring_type)
        
        # The final classifier is a standard linear layer
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride, ring_size, ring_type):
        """Helper function to create a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, ring_size, ring_type))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initializes weights for BatchNorm and Linear layers for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Kaiming initialization is well-suited for layers followed by ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)) # More robust than avg_pool2d
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- Factory Functions to create specific model configurations ---

def ResNet20(num_classes=10, ring_size=8, ring_type='triangle'):
    """Creates a 20-layer ResNet model with Ring Quantization for CIFAR-10."""
    return RingResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, 
                     ring_size=ring_size, ring_type=ring_type)

def ResNet32(num_classes=10, ring_size=8, ring_type='triangle'):
    """Creates a 32-layer ResNet model with Ring Quantization for CIFAR-10."""
    return RingResNet(BasicBlock, [5, 5, 5], num_classes=num_classes,
                     ring_size=ring_size, ring_type=ring_type)

def ResNet44(num_classes=10, ring_size=8, ring_type='triangle'):
    """Creates a 44-layer ResNet model with Ring Quantization for CIFAR-10."""
    return RingResNet(BasicBlock, [7, 7, 7], num_classes=num_classes,
                     ring_size=ring_size, ring_type=ring_type)