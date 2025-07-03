# example.py

"""
Example Usage: Loading a Pretrained Model for Inference

This script demonstrates how to:
1. Load a pretrained Ring Quantization model.
2. Load and preprocess a single image.
3. Perform inference to get a class prediction.
"""
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Import your model definition
from ring_quantization.models import ResNet20

# --- Configuration ---
# Use one of your best pretrained models
CHECKPOINT_PATH = './pretrained/best_resnet20_ring_ring8.pth' 
RING_SIZE = 8
MODEL_ARCH = 'resnet20'

# A random image URL from the web to test inference
IMAGE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog1.png'

# CIFAR-10 class names
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    print("="*80)
    print("Example: Inference with a Pretrained Ring Quantization Model")
    print("="*80)

    # 1. Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Create model architecture
    print(f"\nLoading model architecture: {MODEL_ARCH} with ring size {RING_SIZE}...")
    model = ResNet20(ring_size=RING_SIZE, num_classes=10).to(device)

    # 3. Load pretrained weights
    print(f"Loading pretrained weights from: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        print("Please make sure you have the pretrained models in the './pretrained' directory.")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")
        return

    print("Model loaded successfully!")

    # 4. Load and preprocess the image
    print(f"\nDownloading and preprocessing image from URL...")
    try:
        response = requests.get(IMAGE_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to download or open image: {e}")
        return
    
    # Define the same transformations as used for testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device) # Create a mini-batch as expected by the model

    # 5. Perform inference
    print("Performing inference...")
    with torch.no_grad():
        output = model(input_batch)

    # 6. Get prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.max(probabilities, 0)
    
    predicted_class = CIFAR10_CLASSES[top_catid]
    
    print("\n" + "="*80)
    print("🎉 Inference Result:")
    print(f"   Predicted Class: '{predicted_class}'")
    print(f"   Confidence: {top_prob.item() * 100:.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()