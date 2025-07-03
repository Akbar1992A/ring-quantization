#!/usr/bin/env python3
"""
Ring Quantization Unified Training Script

This script provides a single interface for training and evaluating
both ring-quantized and full-precision (FP32) models on CIFAR-10.

Examples:
    # Train ResNet-20 with 3-bit weights (ring_size=8)
    $ python train.py --model resnet20 --quantization ring --ring-size 8

    # Train a standard, full-precision FP32 ResNet-20 for baseline
    $ python train.py --model resnet20 --quantization none
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

# Import our model libraries
from ring_quantization.models import ResNet20, ResNet32, ResNet44
from ring_quantization.models_fp32 import ResNet20_FP32, ResNet32_FP32, ResNet44_FP32


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior on GPU (can be slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    Prepares and returns CIFAR-10 train and test dataloaders
    with standard augmentations.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def get_model(args):
    """Factory function to create a model instance based on arguments."""
    if args.quantization == 'ring':
        print(f"Loading Ring Quantization model: {args.model}")
        models = {'resnet20': ResNet20, 'resnet32': ResNet32, 'resnet44': ResNet44}
        if args.model not in models:
            raise ValueError(f"Unknown model for ring quantization: {args.model}")
        return models[args.model](ring_size=args.ring_size, ring_type=args.ring_type)
        
    elif args.quantization == 'none':
        print(f"Loading Full-Precision (FP32) model: {args.model}")
        models_fp32 = {'resnet20': ResNet20_FP32, 'resnet32': ResNet32_FP32, 'resnet44': ResNet44_FP32}
        if args.model not in models_fp32:
            raise ValueError(f"Unknown FP32 model: {args.model}")
        return models_fp32[args.model]()
        
    else:
        raise ValueError(f"Unknown quantization type: {args.quantization}. Use 'ring' or 'none'.")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Trains the model for one epoch."""
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        
        optimizer.step()
        
        # Clamp positions only for ring-quantized models
        if args.quantization == 'ring':
            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'positions'):
                        module.positions.data.clamp_(0, 1)
                    if hasattr(module, 'bias_positions') and module.bias_positions is not None:
                        module.bias_positions.data.clamp_(0, 1)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print(f'  Epoch {epoch} [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    return train_loss / len(train_loader), 100. * correct / total


def test_epoch(model, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return test_loss / len(test_loader), 100. * correct / total


def main(args):
    """Main function to run the training and evaluation loop."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Print header
    print("="*80)
    print("Initializing Ring Quantization Training")
    print(f"  - Model: {args.model}")
    print(f"  - Quantization: {args.quantization}")
    if args.quantization == 'ring':
        print(f"  - Ring Size: {args.ring_size} ({np.log2(args.ring_size):.1f} bits)")
        print(f"  - Ring Type: {args.ring_type}")
    print(f"  - Epochs: {args.epochs}")
    print("="*80)
    
    # Setup device, data, model
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}\n")
    train_loader, test_loader = get_cifar10_dataloaders(args.batch_size, args.num_workers)
    model = get_model(args).to(device)
    
    # Define run name for saving checkpoints
    run_name = f'{args.model}_{args.quantization}'
    if args.quantization == 'ring':
        run_name += f'_ring{args.ring_size}'
    
    print(f"Run name: {run_name}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    
    # Main training loop
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save the best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.save_dir, f'best_{run_name}.pth')
                torch.save({'model_state_dict': model.state_dict(), 'best_accuracy': best_accuracy}, checkpoint_path)
                print(f"🏆 New best model saved to {checkpoint_path} with accuracy {best_accuracy:.2f}%")
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{args.epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best: {best_accuracy:.2f}%')
        print(f'  Epoch Time: {epoch_time:.2f}s')
        print("-"*80)
    
    print(f"\n🎉 Training finished for {run_name}!")
    print(f"   Best Test Accuracy: {best_accuracy:.2f}%\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ring Quantization Unified Training Script')
    
    # --- Primary Arguments ---
    parser.add_argument('--quantization', type=str, default='ring', choices=['ring', 'none'],
                        help='Type of model: "ring" for our method, "none" for FP32 baseline.')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet32', 'resnet44'],
                        help='Model architecture to train.')
    
    # --- Ring Quantization Specific Arguments ---
    parser.add_argument('--ring-size', type=int, default=8,
                        help='Number of values in the ring for quantization (e.g., 8 for 3-bit).')
    parser.add_argument('--ring-type', type=str, default='triangle', choices=['triangle', 'sin'],
                        help='Geometric shape of the ring values.')
    
    # --- General Training Arguments ---
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--schedule', nargs='+', type=int, default=[100, 150], help='Epochs at which to drop learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor.')
    
    # --- Optimizer Arguments ---
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 penalty.')
    parser.add_argument('--clip-grad', type=float, default=5.0, help='Max norm for gradient clipping (0 to disable).')
    
    # --- System and I/O Arguments ---
    parser.add_argument('--num-workers', type=int, default=2, help='Number of subprocesses for data loading.')
    parser.add_argument('--log-interval', type=int, default=100, help='Log training status every N batches.')
    parser.add_argument('--save-dir', type=str, default='./pretrained', help='Directory to save best models.')
    parser.add_argument('--no-save-model', action='store_true', default=False, help='Disable saving the best model.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training even if available.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    
    # Post-processing for args
    args.save_model = not args.no_save_model
    
    main(args)