import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import logging
import sys
from pathlib import Path
from PIL import Image
from torchvision import models
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_style_transfer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("StyleTransferTraining")

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleTransferModel(nn.Module):
    """Define the Style Transfer model architecture based on VGG-19"""

    def __init__(self):
        super(StyleTransferModel, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = vgg[:21]  # Use the first 21 layers of VGG-19

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 4:  # Relu5 (content representation)
                content_feature = x
            elif i == 9:  # Relu10 (style representation)
                style_feature = x
            features.append(x)
        return content_feature, style_feature


class StyleLoss(nn.Module):
    """Computes the loss for style transfer"""
    
    def __init__(self, target, style_weight, content_weight):
        super(StyleLoss, self).__init__()
        self.target = target
        self.style_weight = style_weight
        self.content_weight = content_weight

    def forward(self, target):
        content_loss = torch.nn.functional.mse_loss(target[0], self.target[0])
        style_loss = torch.nn.functional.mse_loss(target[1], self.target[1])
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss


def load_image(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    """Load and preprocess an image"""
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)


def train_style_transfer(content_image: Path, style_image: Path, output_dir: Path, epochs: int = 500, learning_rate: float = 0.003):
    """Train the style transfer model"""

    # Prepare the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # Scale image to [0, 255]
    ])
    
    # Load content and style images
    content_image = load_image(content_image, transform)
    style_image = load_image(style_image, transform)

    # Initialize the model and loss function
    model = StyleTransferModel().to(device)
    target_image = content_image.clone().requires_grad_(True)

    # Define optimizer and loss function
    optimizer = optim.Adam([target_image], lr=learning_rate)
    style_transfer_loss = StyleLoss([content_image, style_image], style_weight=1e6, content_weight=1e0)

    # Train loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Get content and style features from the target image
        content_feature, style_feature = model(target_image)

        # Compute the loss
        loss = style_transfer_loss([content_feature, style_feature])

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            logger.info(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

        # Save intermediate results
        if epoch % 50 == 0:
            output_image = target_image.clone().detach().cpu()
            output_image = output_image.squeeze(0).clamp(0, 255).byte()
            save_image = transforms.ToPILImage()(output_image)
            save_image.save(output_dir / f"output_epoch_{epoch}.png")

    logger.info(f"Style Transfer training completed! Final output saved at {output_dir}")
    return target_image


def main():
    """CLI to trigger the training of the style transfer model"""

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a Neural Style Transfer Model")
    parser.add_argument('-c', '--content_image', type=Path, required=True, help="Path to the content image")
    parser.add_argument('-s', '--style_image', type=Path, required=True, help="Path to the style image")
    parser.add_argument('-o', '--output_dir', type=Path, default=Path("style_transfer_results"), help="Directory to save results")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs to train the model")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="Learning rate for the optimizer")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not args.output_dir.exists():
        os.makedirs(args.output_dir)

    # Start training
    trained_image = train_style_transfer(args.content_image, args.style_image, args.output_dir, args.epochs, args.learning_rate)
    
    # Save the final output image
    final_image = trained_image.clone().detach().cpu()
    final_image = final_image.squeeze(0).clamp(0, 255).byte()
    save_final_image = transforms.ToPILImage()(final_image)
    save_final_image.save(args.output_dir / "final_output.png")
    logger.info(f"Final output image saved at {args.output_dir / 'final_output.png'}")

if __name__ == "__main__":
    main()
