"""
AikoInfinity 2.0 Style Transfer Model - AI Engine
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import logging
import os
import sys
from pathlib import Path
from aikoenv import AikoVenvModel, AikoPreprocessor  # Custom Aiko components
from aiko_utils import get_gpu_status  # Custom GPU utilities

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("style_transfer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AikoStyleTransfer")


class StyleTransferModel(nn.Module):
    """
    Class for implementing the style transfer using a pre-trained neural network.
    """

    def __init__(self, style_image: Path, config: dict):
        """Initializes the style transfer model with the given style image."""
        super(StyleTransferModel, self).__init__()

        # Load style image and model configurations
        self.config = config
        self.style_image = cv2.imread(str(style_image))
        self.device = self._select_device()
        self.model = self._initialize_model()
        self.preprocessor = AikoPreprocessor()

        logger.info(f"Style transfer model initialized on {self.device}")

    def _select_device(self) -> torch.device:
        """Selects device for model execution (GPU/CPU)."""
        if self.config['use_cuda'] and torch.cuda.is_available():
            gpu_status = get_gpu_status()  # Custom GPU management
            return torch.device(f"cuda:{gpu_status['optimal_device']}")
        return torch.device("cpu")

    def _initialize_model(self) -> nn.Module:
        """Initializes and loads the pre-trained model for style transfer."""
        try:
            # Load pre-trained VGG19 model and freeze its parameters
            model = torchvision.models.vgg19(pretrained=True).features
            for param in model.parameters():
                param.requires_grad = False
            model = model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading style transfer model: {str(e)}")
            sys.exit(1)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses the input image before passing it to the model."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def _compute_style_loss(self, target_feature, style_feature):
        """Computes the style loss between target and style features."""
        return torch.mean((target_feature - style_feature) ** 2)

    def forward(self, content_image: torch.Tensor) -> torch.Tensor:
        """Apply style transfer on the input content image."""
        # Preprocess style image
        style_image = self.preprocess_image(self.style_image)

        # Extract features of content and style images
        content_features = self._extract_features(content_image)
        style_features = self._extract_features(style_image)

        # Compute loss and apply gradient descent optimization (optional)
        loss = 0
        for content, style in zip(content_features, style_features):
            loss += self._compute_style_loss(content, style)

        # Return the final loss
        return loss

    def _extract_features(self, image: torch.Tensor) -> list:
        """Extracts features from the image using the VGG model."""
        features = []
        x = image
        for layer in self.model.children():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

    def apply_style_transfer(self, content_image: np.ndarray) -> np.ndarray:
        """Apply the style transfer and return the styled image."""
        # Preprocess content image
        content_image_tensor = self.preprocess_image(content_image)

        # Perform style transfer
        styled_image = self.forward(content_image_tensor)

        # Convert the styled image tensor back to an OpenCV image
        styled_image = styled_image.squeeze().cpu().detach().numpy()
        styled_image = np.transpose(styled_image, (1, 2, 0))
        styled_image = (styled_image * 255).astype(np.uint8)

        logger.info("Style transfer applied successfully.")
        return styled_image


def main():
    """Command-line interface for executing the style transfer model."""

    parser = argparse.ArgumentParser(description='AikoInfinity 2.0 Style Transfer')
    parser.add_argument('-c', '--content', type=Path, required=True,
                        help='Path to the content image')
    parser.add_argument('-s', '--style', type=Path, required=True,
                        help='Path to the style image')
    parser.add_argument('-o', '--output', type=Path, default=Path("styled_output"),
                        help='Output directory')
    parser.add_argument('--model', type=Path, default=Path("models/aiko_style_transfer.pth"),
                        help='Path to the pre-trained style transfer model')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable CUDA acceleration if available')

    args = parser.parse_args()

    config = {
        'model_path': args.model,
        'use_cuda': args.cuda
    }

    # Initialize Style Transfer Model
    style_transfer = StyleTransferModel(style_image=args.style, config=config)

    # Read the content image
    content_image = cv2.imread(str(args.content))
    if content_image is None:
        logger.error(f"Failed to read content image: {args.content}")
        sys.exit(1)

    # Apply style transfer
    styled_image = style_transfer.apply_style_transfer(content_image)

    # Save the styled image
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "styled_image.jpg"
    cv2.imwrite(str(output_file), styled_image)

    logger.info(f"Styled image saved to: {output_file}")


if __name__ == "__main__":
    main()
