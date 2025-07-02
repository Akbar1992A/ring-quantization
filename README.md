# Ring Quantization: 2-bit Neural Networks with 89% Accuracy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15788294.svg)](https://doi.org/10.5281/zenodo.15788294)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Ring Quantization: Achieving 89% Accuracy on CIFAR-10 with 2-bit Neural Networks"**

## 🚀 Key Results

- **89.99%** accuracy on CIFAR-10 with **3-bit** weights (ResNet-20)
- **89.27%** accuracy on CIFAR-10 with **2-bit** weights (ResNet-20)
- **16× compression** with minimal accuracy loss
- **State-of-the-art** results for extreme quantization

## 📊 Comparison with Existing Methods

| Method | Bits | CIFAR-10 Accuracy | Compression |
|--------|------|-------------------|-------------|
| Full Precision | 32 | 91.5% | 1× |
| INT8 Quantization | 8 | 91.0% | 4× |
| INT4 Quantization | 4 | 88.0% | 8× |
| **Ring-8 (Ours)** | **3** | **89.99%** | **10.67×** |
| **Ring-4 (Ours)** | **2** | **89.27%** | **16×** |

## 🎯 What is Ring Quantization?

Ring Quantization is a novel neural network compression technique where:
- Each weight is constrained to move along a predefined "ring" of values
- Instead of learning weight values, we learn positions on these rings
- Training is performed by updating positions, not values
- Weights are interpolated using Gaussian kernels for differentiability

## 🛠️ Installation

```bash
git clone https://github.com/Akbar1992A/ring-quantization.git
cd ring-quantization
pip install -r requirements.txt
```

## 🏃 Quick Start

### Train ResNet-20 with 2-bit weights on CIFAR-10:
```python
python train_cifar10.py --model resnet20 --ring-size 4 --epochs 200
```

### Train with 3-bit weights:
```python
python train_cifar10.py --model resnet20 --ring-size 8 --epochs 200
```

### Evaluate pretrained model:
```python
python evaluate.py --checkpoint pretrained/resnet20_ring8.pth
```

## 📁 Repository Structure

```
ring-quantization/
├── models/
│   ├── ring_layers.py      # Core ring quantization layers
│   ├── resnet_ring.py      # ResNet with ring weights
│   └── utils.py            # Helper functions
├── experiments/
│   ├── mnist/              # MNIST experiments
│   ├── cifar10/            # CIFAR-10 experiments
│   └── ablations/          # Ablation studies
├── pretrained/             # Pretrained models
├── train_cifar10.py        # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Dependencies
```

## 📈 Reproduce Our Results

### CIFAR-10 with ResNet-20 (3-bit):
```bash
python train_cifar10.py \
    --model resnet20 \
    --ring-size 8 \
    --ring-type triangle \
    --epochs 200 \
    --lr 0.1 \
    --schedule 100 150 \
    --batch-size 128
```

Expected result: ~89.99% accuracy

### CIFAR-10 with ResNet-20 (2-bit):
```bash
python train_cifar10.py \
    --model resnet20 \
    --ring-size 4 \
    --ring-type triangle \
    --epochs 200 \
    --lr 0.1 \
    --schedule 100 150 \
    --batch-size 128
```

Expected result: ~89.27% accuracy

## 🔬 How It Works

```python
import torch
import torch.nn as nn

class RingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ring_size=8):
        super().__init__()
        # Create ring of values (triangle wave)
        t = torch.linspace(0, 2*torch.pi, ring_size)
        self.ring = 2 * torch.abs(2 * (t/(2*torch.pi) - torch.floor(t/(2*torch.pi) + 0.5))) - 1
        
        # Learn positions on the ring
        self.positions = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
    
    def forward(self, x):
        # Interpolate weights from ring positions
        weights = interpolate_from_ring(self.positions, self.ring)
        return F.conv2d(x, weights)
```

## 📊 Additional Results

### MNIST (95% accuracy with 8 values per weight):
```bash
python train_mnist.py --ring-size 8 --epochs 10
```

### Ablation Studies:
- Triangle rings outperform sinusoidal by 1-2%
- Gaussian interpolation superior to linear
- Ring size 8 optimal for accuracy/compression trade-off

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@article{Xalilov2025ring,
  title={Ring Quantization: Achieving 89% Accuracy on CIFAR-10 with 2-bit Neural Networks},
  author={Xalilov, Akbarali},
  year={2025},
  doi={10.5281/zenodo.15788294}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for extreme model compression
- Thanks to the PyTorch team for the excellent framework

## 📧 Contact

Akbarali Xalilov - [bigdatateg@gmail.com]

Project Link: [https://github.com/username/ring-quantization](https://github.com/Akbar1992A/ring-quantization)

---

**⭐ If you find this work useful, please consider giving it a star!**