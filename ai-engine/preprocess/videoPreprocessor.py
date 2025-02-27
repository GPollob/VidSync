"""
AikoInfinity 2.0 Video Preprocessor - AI Engine
This script handles video preprocessing tasks including frame extraction, resizing, and normalization
to prepare video data for AI models.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("AikoVideoPreprocessor")


class VideoPreprocessor:
    """
    Class for preprocessing video data, including frame extraction, resizing, and normalization.
    """

    def __init__(self, frame_size: Tuple[int, int] = (224, 224), max_frames: int = 100):
        """
        Initializes the video preprocessor with parameters like frame size and maximum number of frames to process.
        
        :param frame_size: Desired frame size (height, width) for the frames.
        :param max_frames: The maximum number of frames to extract from the video.
        """
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet pre-trained normalization
        ])
        logger.info(f"VideoPreprocessor initialized with frame size {frame_size} and max frames {max_frames}.")

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from the video file.
        
        :param video_path: Path to the video file.
        :return: A list of frames (as numpy arrays).
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        frame_count = 0
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frames.append(frame)
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video.")
        return frames

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a single frame to the desired dimensions.
        
        :param frame: The frame to resize.
        :return: Resized frame.
        """
        frame_resized = cv2.resize(frame, self.frame_size)
        logger.debug(f"Resized frame to {self.frame_size}.")
        return frame_resized

    def preprocess_frames(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Preprocess a list of frames by resizing and normalizing.
        
        :param frames: List of frames (as numpy arrays).
        :return: List of preprocessed frames (as tensors).
        """
        preprocessed_frames = []
        for frame in frames:
            frame_resized = self.resize_frame(frame)
            frame_tensor = self.transform(Image.fromarray(frame_resized))  # Convert to PIL Image before transforming
            preprocessed_frames.append(frame_tensor)
        logger.info(f"Preprocessed {len(preprocessed_frames)} frames.")
        return preprocessed_frames

    def preprocess_video(self, video_path: str) -> List[torch.Tensor]:
        """
        Preprocesses a video by extracting frames and performing resizing and normalization.
        
        :param video_path: Path to the video file.
        :return: List of preprocessed frames (as tensors).
        """
        frames = self.extract_frames(video_path)
        preprocessed_frames = self.preprocess_frames(frames)
        return preprocessed_frames

    def encode_video(self, video_path: str) -> torch.Tensor:
        """
        Encodes the video into a tensor format suitable for feeding into models.
        
        :param video_path: Path to the video file.
        :return: A tensor containing the video frames (N, C, H, W) where N = number of frames, C = channels, H = height, W = width.
        """
        preprocessed_frames = self.preprocess_video(video_path)
        video_tensor = torch.stack(preprocessed_frames)  # Stack frames into a single tensor
        logger.debug(f"Encoded video into tensor with shape {video_tensor.shape}.")
        return video_tensor

    def decode_video(self, video_tensor: torch.Tensor) -> List[np.ndarray]:
        """
        Decodes a video tensor back into frames (for visualization or post-processing).
        
        :param video_tensor: The tensor containing video frames.
        :return: A list of decoded frames (as numpy arrays).
        """
        frames = video_tensor.permute(0, 2, 3, 1).numpy()  # Convert tensor to numpy array (N, H, W, C)
        frames = [np.array(frame, dtype=np.uint8) for frame in frames]  # Convert each frame to uint8
        logger.debug(f"Decoded video tensor into {len(frames)} frames.")
        return frames


def test_video_preprocessing():
    """
    Function to test the video preprocessor on a sample video.
    """
    video_path = "sample_video.mp4"  # Provide the path to your video
    preprocessor = VideoPreprocessor(frame_size=(224, 224), max_frames=100)
    
    # Preprocess the video and obtain frames
    video_tensor = preprocessor.encode_video(video_path)
    print("Encoded Video Tensor Shape:", video_tensor.shape)

    # Decode the video back to frames
    frames = preprocessor.decode_video(video_tensor)
    print(f"Decoded {len(frames)} frames from the video.")

    # Optionally visualize or save frames for verification
    for i, frame in enumerate(frames[:5]):  # Visualize first 5 frames
        Image.fromarray(frame).show()
        if i == 4:
            break


if __name__ == "__main__":
    test_video_preprocessing()
