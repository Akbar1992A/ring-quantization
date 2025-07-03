"""
Ring Quantization: A novel approach to extreme neural network compression
"""

from .layers import RingConv2d, RingLinear
from .models import ResNet20, ResNet32, BasicBlock, RingResNet

__version__ = "1.0.0"
__author__ = "Akbarali Xalilov"

__all__ = [
    "RingConv2d",
    "RingLinear", 
    "ResNet20",
    "ResNet32",
    "BasicBlock",
    "RingResNet"
]
