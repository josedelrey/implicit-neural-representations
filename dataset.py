import os
import numpy as np
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torch
from torch.utils.data import Dataset


def get_mgrid(width, height, dim=2):
    '''Generates a flattened grid of (x,y) coordinates in a range of -1 to 1,
    adjusted for non-square aspect ratios.
    
    width: int  (image width)
    height: int (image height)
    dim: int (default=2 for 2D images)
    '''
    x = torch.linspace(-1, 1, steps=width)  # Keep width scaling
    y = torch.linspace(-1, 1, steps=height)  # Keep height scaling

    mgrid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)  # Note 'ij' indexing
    mgrid = mgrid.reshape(-1, dim)  # Flatten
    return mgrid


def get_image_tensor(sidelength, path, channels=1):
    # Open image and convert based on channels value
    if channels == 3:
        img = Image.open(path).convert("RGB")
    elif channels == 1:
        img = Image.open(path).convert("L")
    else:
        raise ValueError("channels must be either 1 (greyscale) or 3 (RGB)")
    
    width, height = img.size

    # Compute new width and height maintaining aspect ratio
    if width >= height:
        new_width = sidelength
        new_height = int((height / width) * sidelength)
    else:
        new_height = sidelength
        new_width = int((width / height) * sidelength)

    # Use different normalization parameters based on number of channels
    if channels == 3:
        transform = Compose([
            Resize((new_height, new_width)),  # Maintain aspect ratio
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = Compose([
            Resize((new_height, new_width)),  # Maintain aspect ratio
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])

    img = transform(img)
    
    return img, new_width, new_height


class ImageDataset(Dataset):
    def __init__(self, sidelength, path, channels=1):
        super().__init__()
        # Get the image tensor along with its new dimensions
        img, self.width, self.height = get_image_tensor(sidelength, path, channels)

        # Permute to (H, W, C) and flatten pixels; uses the proper number of channels based on is_rgb
        self.pixels = img.permute(1, 2, 0).view(-1, channels)
        self.coords = get_mgrid(self.width, self.height)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels
