import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import numpy as np
import cv2
from typing import Tuple

class PixelDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int]):
        # Read and normalize image
        image = imageio.imread(image_path)[..., :3] / 255.0
        image = cv2.resize(image, img_wh)  # (H, W, 3)
        
        H, W = image.shape[:2]

        # Create normalized coordinate vectors for x and y in range [-1, 1]
        xs = torch.linspace(-1, 1, steps=W)
        ys = torch.linspace(-1, 1, steps=H)

        # Create meshgrid. Using 'ij' indexing gives grid_y of shape (H, W) and grid_x of shape (H, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        # Stack to form a grid with shape (H, W, 2) where the last dimension is (x, y)
        uv = torch.stack((grid_x, grid_y), dim=-1) # shape: (H, W, 2)
        
        # Convert image to tensor (shape: H x W x 3)
        rgb = torch.FloatTensor(image)
        
        # Flatten the first two dimensions (height, width) into one dimension
        self.uv = uv.reshape(-1, uv.shape[-1])   # shape: (H*W, 2)
        self.rgb = rgb.reshape(-1, rgb.shape[-1])  # shape: (H*W, 3)
    
    def __len__(self):
        return self.uv.shape[0]
    
    def __getitem__(self, idx: int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}
