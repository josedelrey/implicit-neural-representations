import imageio.v2 as imageio
import cv2
import torch
from torch.utils.data import Dataset


class PixelDataset(Dataset):
    """
    A PyTorch Dataset for loading an image and generating normalized pixel coordinate (uv)
    and corresponding RGB value pairs.

    The image is read from a file and normalized to the range [0, 1]. UV coordinates are generated
    for each pixel in the image in the range [-1, 1] for both x and y axes. The UV coordinates and
    RGB values are then flattened into 1D arrays, where each entry corresponds to a single pixel.

    Args:
        image_path (str): Path to the image file.

    Attributes:
        uv (torch.Tensor): A tensor of shape (H*W, 2) containing normalized pixel coordinates,
            where H and W are the height and width of the image.
        rgb (torch.Tensor): A tensor of shape (H*W, 3) containing the corresponding normalized RGB values.
    """
    def __init__(self, image_path: str):
        # Read and normalize image
        image = imageio.imread(image_path)[..., :3] / 255.0
        
        H, W = image.shape[:2]

        # Create normalized coordinate vectors for x and y in the range [-1, 1]
        xs = torch.linspace(-1, 1, steps=W)
        ys = torch.linspace(-1, 1, steps=H)

        # Create grid
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        # Stack to form a grid with shape (H, W, 2)
        uv = torch.stack((grid_x, grid_y), dim=-1)
        
        # Convert image to tensor
        rgb = torch.FloatTensor(image)
        
        # Flatten the first two dimensions (height, width) into one dimension
        self.uv = uv.reshape(-1, uv.shape[-1])  # shape: (H*W, 2)
        self.rgb = rgb.reshape(-1, rgb.shape[-1])  # shape: (H*W, 3)
    
    def __len__(self) -> int:
        return self.uv.shape[0]
    
    def __getitem__(self, idx: int) -> dict:
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}
