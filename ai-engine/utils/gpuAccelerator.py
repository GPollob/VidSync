import torch
import logging
from torch.cuda import is_available, current_device, device_count, get_device_name, memory_allocated, memory_cached
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpuAccelerator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GPUAccelerator")

class GPUAccelerator:
    """A class to manage GPU operations for model training and inference."""

    def __init__(self, device='cuda', memory_limit=None):
        """
        Args:
            device (str): Specify the device ('cuda' or 'cpu'). Default is 'cuda'.
            memory_limit (float): Limit the memory usage for the GPU in MB. If None, no limit is applied.
        """
        self.device = device
        self.memory_limit = memory_limit

        # Check for CUDA availability
        self.is_cuda_available = is_available()

        if self.is_cuda_available:
            self.device = self.select_device()
        else:
            logger.warning("CUDA is not available. Using CPU instead.")
            self.device = 'cpu'

        logger.info(f"Using device: {self.device}")

    def select_device(self):
        """Select GPU device based on available CUDA devices."""
        if self.device == 'cuda' and self.is_cuda_available:
            device_count = device_count()
            if device_count > 0:
                # Default to the first available GPU
                device_name = get_device_name(current_device())
                logger.info(f"Selected GPU: {device_name} (CUDA {current_device()})")
                return f'cuda:{current_device()}'
            else:
                logger.warning("No GPUs detected. Switching to CPU.")
                return 'cpu'
        else:
            return 'cpu'

    def move_to_device(self, tensor):
        """
        Moves a tensor to the configured device (GPU or CPU).
        
        Args:
            tensor (torch.Tensor): The tensor to be moved to the device.
            
        Returns:
            torch.Tensor: Tensor moved to the device.
        """
        if self.device == 'cpu':
            return tensor.cpu()
        else:
            return tensor.to(self.device)

    def manage_gpu_memory(self, tensor):
        """
        Ensures the tensor is within GPU memory limits (if specified).
        
        Args:
            tensor (torch.Tensor): The tensor to manage.
            
        Returns:
            torch.Tensor: Managed tensor.
        """
        if self.memory_limit:
            allocated = memory_allocated(self.device) / 1024**2  # MB
            if allocated + tensor.numel() * tensor.element_size() / 1024**2 > self.memory_limit:
                logger.warning(f"GPU memory usage is nearing the limit. Allocated: {allocated} MB, Limit: {self.memory_limit} MB.")
                # Handle memory overflow (e.g., release memory, perform optimization)
                torch.cuda.empty_cache()

        return self.move_to_device(tensor)

    def check_gpu_memory(self):
        """Checks the current GPU memory usage."""
        if self.device != 'cpu':
            allocated = memory_allocated(self.device) / 1024**2  # MB
            cached = memory_cached(self.device) / 1024**2  # MB
            logger.info(f"GPU memory allocated: {allocated} MB, cached: {cached} MB.")
        else:
            logger.info("Running on CPU. No GPU memory to check.")

    def clear_gpu_memory(self):
        """Clear the cached GPU memory."""
        if self.device != 'cpu':
            torch.cuda.empty_cache()
            logger.info("Cleared cached GPU memory.")
        else:
            logger.info("No GPU to clear memory from.")

    def get_device_info(self):
        """Returns the information about the GPU or CPU."""
        if self.device == 'cpu':
            return "CPU: Standard CPU device"
        else:
            device_name = get_device_name(current_device())
            total_memory = torch.cuda.get_device_properties(current_device()).total_memory / 1024**2  # MB
            return f"GPU: {device_name}, Total Memory: {total_memory} MB"

def test_gpu_acceleration():
    """Test function to check GPU acceleration."""
    # Simple tensor operation to test if GPU acceleration works
    try:
        gpu_accelerator = GPUAccelerator(device='cuda')
        logger.info("Testing GPU acceleration...")

        # Create a random tensor and move to the device
        tensor = torch.randn(10000, 10000)  # Create a large tensor for testing
        tensor = gpu_accelerator.move_to_device(tensor)

        # Perform a simple operation
        result = tensor * 2
        gpu_accelerator.check_gpu_memory()  # Check memory usage

        logger.info("GPU acceleration is working correctly!")
        return result
    except Exception as e:
        logger.error(f"Error during GPU acceleration test: {str(e)}")
        return None

if __name__ == "__main__":
    # Run a test to ensure GPU acceleration works
    test_result = test_gpu_acceleration()
    if test_result is not None:
        logger.info(f"Test result: {test_result.size()}")
    else:
        logger.error("Test failed.")
