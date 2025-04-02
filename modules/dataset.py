import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image


def get_mgrid(width, height, dim=2):
    """
    Generates a flattened grid of (x, y) coordinates in the range of -1 to 1.

    Args:
        width (int): The width of the grid (number of x-coordinates).
        height (int): The height of the grid (number of y-coordinates).
        dim (int, optional): Dimensionality of the coordinates.

    Returns:
        torch.Tensor: A tensor of shape (width*height, dim) containing the grid coordinates.
    """
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_image_tensor(sidelength, path, channels=1):
    """
    Loads an image, resizes it while maintaining aspect ratio, and normalizes the image tensor.

    Args:
        sidelength (int): The target size for the longer side of the image.
        path (str): The file path to the image.
        channels (int, optional): The number of channels desired in the output image 
                                  (1 for greyscale, 3 for RGB).

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The normalized image tensor.
            - int: The new width of the image.
            - int: The new height of the image.
    """
    if channels == 3:
        img = Image.open(path).convert("RGB")
    elif channels == 1:
        img = Image.open(path).convert("L")
    else:
        raise ValueError("Channels must be either 1 (greyscale) or 3 (RGB)")

    width, height = img.size

    # Compute new width and height while maintaining the aspect ratio
    if width >= height:
        new_width = sidelength
        new_height = int((height / width) * sidelength)
    else:
        new_height = sidelength
        new_width = int((width / height) * sidelength)

    # Define transforms with appropriate normalization parameters
    if channels == 3:
        transform = Compose([
            Resize((new_height, new_width)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = Compose([
            Resize((new_height, new_width)),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])

    img = transform(img)
    
    return img, new_width, new_height


class ImageDataset(Dataset):
    """
    Custom dataset for processing an image. This dataset loads an image, resizes it, 
    and generates a grid of coordinates along with the corresponding pixel values.
    """
    def __init__(self, sidelength, path, channels=1):
        """
        Initializes the ImageDataset.

        Args:
            sidelength (int): The target size for the longer side of the image.
            path (str): The file path to the image.
            channels (int, optional): The number of channels (1 for greyscale, 3 for RGB).
        """
        super().__init__()
        # Get the image tensor along with its new dimensions
        img, self.width, self.height = get_image_tensor(sidelength, path, channels)

        # Permute to (H, W, C) and flatten pixels
        self.pixels = img.permute(1, 2, 0).view(-1, channels)

        # Generate grid coordinates for each pixel
        self.coords = get_mgrid(self.width, self.height)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Retrieves the grid coordinates and pixel values for the image.

        Args:
            idx (int): Index of the item to retrieve. Only index 0 is valid.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The grid coordinates of shape (width*height, 2).
                - torch.Tensor: The pixel values of shape (width*height, channels).
        """
        if idx > 0:
            raise IndexError("Index out of range. This dataset only contains one item.")
        return self.coords, self.pixels
