# evaluate.py

#!/usr/bin/env python3
"""
Evaluation Script for Ring Quantization Models

This script loads a pretrained model checkpoint and evaluates its accuracy
on the CIFAR-10 test set.

Example:
    # Evaluate a 3-bit ResNet-20 model
    $ python evaluate.py --model resnet20 --quantization ring --ring-size 8 --checkpoint ./pretrained/best_resnet20_ring_ring8.pth

    # Evaluate an FP32 ResNet-20 baseline
    $ python evaluate.py --model resnet20 --quantization none --checkpoint ./pretrained/best_resnet20_fp32.pth
"""

import argparse
import torch
from train import get_cifar10_dataloaders, get_model, test_epoch  # Re-use functions from train.py
import torch.nn as nn

def main(args):
    """Main evaluation function."""
    print("="*80)
    print("Initializing Model Evaluation")
    print(f"  - Model: {args.model}")
    print(f"  - Quantization: {args.quantization}")
    if args.quantization == 'ring':
        print(f"  - Ring Size: {args.ring_size}")
    print(f"  - Checkpoint: {args.checkpoint}")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model architecture
    model = get_model(args).to(device)

    # Load the checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully from checkpoint.")
        if 'best_accuracy' in checkpoint:
            print(f"Checkpoint reported best accuracy: {checkpoint['best_accuracy']:.2f}%")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Get test dataloader
    _, test_loader = get_cifar10_dataloaders(args.batch_size, args.num_workers)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    print("\nRunning evaluation on the test set...")
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

    print("\n" + "="*80)
    print("Evaluation Complete")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_acc:.2f}%")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ring Quantization Evaluation Script')
    
    # Model arguments (must match the trained model)
    parser.add_argument('--quantization', type=str, required=True, choices=['ring', 'none'],
                        help='Type of the model to evaluate.')
    parser.add_argument('--model', type=str, required=True, choices=['resnet20', 'resnet32', 'resnet44'],
                        help='Model architecture.')
    parser.add_argument('--ring-size', type=int, default=8,
                        help='Ring size (required for ring quantization models).')
    
    # Checkpoint argument
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint .pth file.')

    # Dataloader arguments
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers.')

    args = parser.parse_args()
    main(args)