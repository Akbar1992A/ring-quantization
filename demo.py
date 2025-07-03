# demo.py

"""
Quick Installation and Sanity Check Demo

This script performs a quick sanity check to ensure that the core components
of the Ring Quantization library are installed and working correctly.

It does the following:
1. Imports necessary libraries (torch, etc.).
2. Creates a 2-bit ResNet-20 model instance.
3. Performs a forward pass with a random tensor.

If this script runs without errors, your environment is likely set up correctly.
"""

import torch
from ring_quantization.models import ResNet20

def run_demo():
    print("="*50)
    print("Ring Quantization Demo: Sanity Check")
    print("="*50)

    try:
        # 1. Check if PyTorch and CUDA are available
        print(f"PyTorch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 2. Create a model instance (smallest and fastest one)
        print("\nAttempting to create a 2-bit (Ring-4) ResNet-20 model...")
        model = ResNet20(ring_size=4).to(device)
        print("Model created successfully!")

        # 3. Perform a forward pass with a random tensor
        print("\nPerforming a forward pass with a random input tensor...")
        # Create a random tensor simulating a batch of 1 image from CIFAR-10
        random_input = torch.randn(1, 3, 32, 32).to(device)
        
        with torch.no_grad():
            output = model(random_input)
        
        print("Forward pass completed successfully!")
        print(f"Output shape: {output.shape}")
        print(f"Output values (logits): {output.squeeze().cpu().numpy()}")

        print("\n🎉 Demo completed successfully! Your environment seems ready.")
        print("="*50)

    except Exception as e:
        print("\n❌ An error occurred during the demo.")
        print("Error details:", e)
        print("\nPlease ensure all dependencies from requirements.txt are installed.")
        print("="*50)

if __name__ == '__main__':
    run_demo()