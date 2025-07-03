"""
Ring Quantization Layers

This module contains the core layers for Ring Quantization, a novel method
for neural network compression. The key idea is to learn continuous positions
on a predefined ring of values, rather than learning the weight values directly.

Layers:
- RingConv2d: A convolutional layer with ring-quantized weights.
- RingLinear: A fully-connected (linear) layer with ring-quantized weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class _RingQuantizer(nn.Module):
    """
    A helper base class that handles the ring creation and interpolation logic.
    This avoids code duplication between RingConv2d and RingLinear.
    """
    def __init__(self, ring_size, ring_type):
        super().__init__()
        self.ring_size = ring_size
        
        # Create the ring template and register it as a non-trainable buffer
        ring_values = self._create_ring(ring_type, ring_size)
        self.register_buffer('ring_template', ring_values)

    def _create_ring(self, ring_type, size):
        """Creates the predefined ring of values with a specified shape."""
        # Note: We don't specify the device here, it will be inferred from the model's device later.
        t = torch.linspace(0, 2 * np.pi, size)
        if ring_type == 'triangle':
            # A simple triangle wave from -1 to 1, provides sharp decision boundaries.
            return 2 * torch.abs(2 * (t / (2 * np.pi) - torch.floor(t / (2 * np.pi) + 0.5))) - 1
        elif ring_type == 'sin':
            # A smooth sinusoidal wave from -1 to 1.
            return torch.sin(t)
        else:
            raise ValueError(f"Unknown ring type: {ring_type}")

    def _interpolate(self, positions):
        """
        Differentiably interpolates values from the ring using Gaussian kernels.
        This is the core of the method, allowing gradients to flow to the positions.
        """
        # Ensure the ring_template is on the same device as the positions
        ring_template = self.ring_template.to(positions.device)

        # Scale positions (0-1) to ring indices
        ring_positions = positions * self.ring_size
        indices = torch.arange(self.ring_size, device=positions.device, dtype=torch.float32)
        
        # Compute circular distances (the shortest path on the ring)
        distances = torch.abs(ring_positions.unsqueeze(-1) - indices.unsqueeze(0))
        circular_distances = torch.min(distances, self.ring_size - distances)
        
        # Use an adaptive sigma for the Gaussian kernel. A smaller sigma for denser rings.
        sigma = max(0.5, self.ring_size / 32.0)
        
        # Calculate soft attention-like weights based on distance
        attention_weights = F.softmax(-0.5 * (circular_distances / sigma) ** 2, dim=-1)
        
        # Compute the final value as a weighted average of ring template values
        values = torch.matmul(attention_weights, ring_template)
        return values


class RingConv2d(_RingQuantizer):
    """Convolutional layer with ring-based weight quantization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, 
                 ring_size=8, ring_type='triangle', bias=True):
        super().__init__(ring_size, ring_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Learnable parameters: continuous positions on the ring [0, 1]
        n_weights = out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.positions = nn.Parameter(torch.rand(n_weights))
        
        if bias:
            self.bias_positions = nn.Parameter(torch.rand(out_channels))
        else:
            self.register_parameter('bias_positions', None)
    
    def forward(self, x):
        # Interpolate weights from their positions on the ring
        effective_weights = self._interpolate(self.positions)
        effective_weights = effective_weights.view(self.out_channels, self.in_channels, *self.kernel_size)
        
        effective_bias = None
        if self.use_bias:
            effective_bias = self._interpolate(self.bias_positions)
        
        return F.conv2d(x, effective_weights, effective_bias, self.stride, self.padding)


class RingLinear(_RingQuantizer):
    """Fully-connected (linear) layer with ring-based weight quantization."""
    def __init__(self, in_features, out_features, ring_size=8, ring_type='triangle', bias=True):
        super().__init__(ring_size, ring_type)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Learnable parameters: continuous positions on the ring [0, 1]
        self.positions = nn.Parameter(torch.rand(out_features, in_features))
        
        if bias:
            self.bias_positions = nn.Parameter(torch.rand(out_features))
        else:
            self.register_parameter('bias_positions', None)

    def forward(self, x):
        # Interpolate weights from their positions
        # We need to interpolate for each weight, so we flatten first
        effective_weights = self._interpolate(self.positions.flatten())
        effective_weights = effective_weights.view(self.out_features, self.in_features)
        
        effective_bias = None
        if self.use_bias:
            effective_bias = self._interpolate(self.bias_positions)
        
        return F.linear(x, effective_weights, effective_bias)