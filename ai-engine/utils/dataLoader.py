import os
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataLoader.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DataLoader")

class CustomImageDataset(data.Dataset):
    """Custom Dataset for loading images and corresponding labels."""
    
    def __init__(self, data_dir, label_file=None, transform=None, image_size=(256, 256)):
        """
        Args:
            data_dir (str): Directory containing images.
            label_file (str): Path to a CSV or JSON file with labels (optional).
            transform (callable, optional): Optional transform to be applied on an image.
            image_size (tuple): Desired output image size for resizing.
        """
        self.data_dir = Path(data_dir)
        self.label_file = label_file
        self.transform = transform
        self.image_size = image_size

        # Load labels (if any)
        if self.label_file:
            self.labels = self.load_labels(self.label_file)
        else:
            self.labels = None
        
        # List all image paths in the directory
        self.image_paths = list(self.data_dir.glob('**/*.jpg')) + list(self.data_dir.glob('**/*.png'))
        
        logger.info(f"Loaded {len(self.image_paths)} images from {data_dir}.")

    def load_labels(self, label_file):
        """Load labels from a CSV or JSON file."""
        if label_file.endswith('.csv'):
            return pd.read_csv(label_file)
        elif label_file.endswith('.json'):
            with open(label_file, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported label file format. Use CSV or JSON.")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Resize image
        image = image.resize(self.image_size)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        # Load label if available
        label = None
        if self.labels is not None:
            img_name = img_path.stem
            label = self.get_label_for_image(img_name)
        
        return image, label
    
    def get_label_for_image(self, image_name):
        """Get label for an image by its filename."""
        if isinstance(self.labels, pd.DataFrame):
            label = self.labels[self.labels['filename'] == image_name].iloc[0]['label']
        elif isinstance(self.labels, dict):
            label = self.labels.get(image_name, None)
        return label

class TextDataset(data.Dataset):
    """Dataset for loading text data and labels."""
    
    def __init__(self, text_data, labels=None, transform=None):
        """
        Args:
            text_data (list or str): List of text samples.
            labels (list or None): List of corresponding labels (optional).
            transform (callable, optional): Optional transform to be applied to text data.
        """
        self.text_data = text_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]

        # Apply transformations if any
        if self.transform:
            text = self.transform(text)

        label = None
        if self.labels is not None:
            label = self.labels[idx]
        
        return text, label

def get_image_transforms():
    """Return standard transformations for image preprocessing."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_text_transform():
    """Return transformations for text data preprocessing (e.g., tokenization)."""
    # This can be a placeholder if using models that require tokenization or embedding
    return lambda x: x.lower()  # A simple example of lowercasing text

def create_dataloader(data_dir, label_file=None, batch_size=32, image_size=(256, 256), shuffle=True):
    """Create a DataLoader for the image dataset."""
    transform = get_image_transforms()
    
    dataset = CustomImageDataset(
        data_dir=data_dir,
        label_file=label_file,
        transform=transform,
        image_size=image_size
    )
    
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    return dataloader

def create_text_dataloader(text_data, labels=None, batch_size=32, shuffle=True):
    """Create a DataLoader for the text dataset."""
    transform = get_text_transform()

    dataset = TextDataset(
        text_data=text_data,
        labels=labels,
        transform=transform
    )

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

def save_processed_data(data, output_path):
    """Save processed data (e.g., images or text) to a file."""
    if isinstance(data, torch.Tensor):
        torch.save(data, output_path)
        logger.info(f"Saved data to {output_path}")
    elif isinstance(data, list):
        with open(output_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")
        logger.info(f"Saved text data to {output_path}")
    else:
        raise ValueError("Unsupported data format for saving")

def load_processed_data(input_path):
    """Load previously processed data from a file."""
    if input_path.endswith('.pt'):
        return torch.load(input_path)
    elif input_path.endswith('.txt'):
        with open(input_path, 'r') as f:
            return f.read().splitlines()
    else:
        raise ValueError("Unsupported file format for loading data")

def visualize_sample(image, title="Sample Image"):
    """Visualize a sample image (for debugging purposes)."""
    import matplotlib.pyplot as plt

    image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage of the dataLoader.py script
    data_dir = "./data/images"
    label_file = "./data/labels.json"
    batch_size = 32
    image_size = (256, 256)

    # Create image dataloader
    dataloader = create_dataloader(data_dir, label_file, batch_size, image_size)

    # Iterate through the dataloader
    for images, labels in dataloader:
        logger.info(f"Batch size: {images.size(0)}, Image size: {images.size(2)}x{images.size(3)}")
        visualize_sample(images[0], "Sample Image")

    # Example usage for text data
    text_data = ["This is a sample text.", "Another example of text."]
    labels = [0, 1]  # Binary classification labels
    text_dataloader = create_text_dataloader(text_data, labels, batch_size)

    for text, label in text_dataloader:
        logger.info(f"Text: {text}, Label: {label}")
