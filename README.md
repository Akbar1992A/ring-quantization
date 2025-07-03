# Ring Quantization: Near-Lossless 2-bit and 3-bit Deep Networks

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15800775.svg)](https://doi.org/10.5281/zenodo.15800775)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official PyTorch implementation for the paper: **"Ring Quantization: Near-Lossless 2-bit and 3-bit Deep Networks"** by Akbarali Xalilov.

This repository contains the full code to reproduce all experiments, along with pretrained models.

---

## 🚀 Key Results & Highlights

Ring Quantization is a novel neural network compression technique that achieves state-of-the-art accuracy at ultra-low bit-widths, demonstrating properties previously thought to be unattainable.

*   **Near-Lossless Performance:** Achieves ~90% accuracy on CIFAR-10 with both 2-bit and 3-bit weights. The accuracy drop from 3-bit to 2-bit is a mere **~0.7%**.
*   **Excellent Scalability & The Depth Synergy Paradox:** Performance is maintained or even *improves* on deeper networks. Our 2-bit ResNet-32 **outperforms** the 2-bit ResNet-20, suggesting a unique synergy between Ring Quantization and model depth.
*   **State-of-the-Art Results:** The demonstrated accuracies surpass existing foundational methods for extreme quantization by a significant margin.
*   **Simple and Robust:** The method is easy to implement and does not require complex, multi-stage training procedures.

### 📊 Final Performance Matrix on CIFAR-10

| Model | Bits | Best Accuracy | Compression |
|:---|:---:|:---:|:---:|
| **ResNet-20** | **3-bit** | **89.99%** | **~10.7×** |
| **ResNet-20** | **2-bit** | **89.27%** | **16×** |
| **ResNet-32** | **3-bit** | **90.01%** | **~10.7×** |
| **ResNet-32** | **2-bit** | **89.29%** | **16×** |
| (FP32 Baseline, ResNet-20) | 32-bit | (91.93%) | 1× |

---

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Akbar1992A/ring-quantization.git
    cd ring-quantization
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) Run the demo to ensure your environment is set up correctly:
    ```bash
    python demo.py
    ```

---

## 🏃‍♀️ Training & Evaluation

This project uses a unified script, `train.py`, for all training operations.

### Reproduce Key Results

**Train ResNet-32 with 2-bit weights (target ~89.29%):**
```bash
python train.py --model resnet32 --quantization ring --ring-size 4 --seed 42
```

**Train ResNet-32 with 3-bit weights (target ~90.01%):**
```bash
python train.py --model resnet32 --quantization ring --ring-size 8 --seed 42
```

**Train the FP32 baseline (target ~91.93%):**
```bash
python train.py --model resnet20 --quantization none --seed 23
```

### Evaluate a Pretrained Model

Use the `evaluate.py` script to validate the accuracy of our provided pretrained models.

```bash
python evaluate.py --model resnet32 --quantization ring --ring-size 4 --checkpoint ./pretrained/best_ResNet32_ring4.pth
```

---

## 🤝 Citation

If you find this work useful in your research, please consider citing our paper. You can use the `CITATION.cff` file for automatic citation generation in GitHub.

```bibtex
@article{Xalilov2025ring,
  author       = {Xalilov, Akbarali},
  title        = {{Ring Quantization: Near-Lossless 2-bit and 3-bit Deep Networks}},
  month        = {jul},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {2.0},
  doi          = {10.5281/zenodo.15800775},
  url          = {https://doi.org/10.5281/zenodo.15800775}
}
```

---

## 🔬 Future Work & Collaboration

The results on CIFAR-10 are extremely promising, but this is just the beginning. As an independent researcher with limited computational resources, I am actively seeking collaborations to test and scale Ring Quantization on larger benchmarks.

Key research directions include:

*   **ImageNet:** Evaluating on ResNet-18/50.
*   **Transformers:** Applying the method to Vision Transformers (ViT) and Large Language Models (LLMs).
*   **Adaptive Rings:** Exploring the potential of learnable ring parameters.

If you are a researcher from an academic lab or an industrial R&D team with available compute, I would be excited to collaborate. Please feel free to reach out via the email on my profile.

---

⭐ **If this work helps you, please consider giving the repository a star!**
