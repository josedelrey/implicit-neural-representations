import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image
import imageio


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


def get_mgrid3d(num_frames, width, height, dim=3):
    """
    Generates a flattened grid of (t, y, x) coordinates for the video.
    The time dimension along with the spatial dimensions are normalized to the range -1 to 1.

    Args:
        num_frames (int): Number of frames in the video.
        width (int): The frame width (number of x-coordinates).
        height (int): The frame height (number of y-coordinates).
        dim (int, optional): Dimensionality of the grid coordinates, should be 3.

    Returns:
        torch.Tensor: A tensor of shape (num_frames * width * height, 3) containing the grid coordinates.
    """
    # Create normalized coordinates for each dimension:
    t = torch.linspace(-1, 1, steps=num_frames)
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    # Create a meshgrid with order (time, y, x). The ordering helps keep consistency with the tensor shape later.
    grid = torch.stack(torch.meshgrid(t, y, x, indexing='ij'), dim=-1)  # shape: (T, H, W, 3)
    grid = grid.reshape(-1, dim)
    return grid


def get_video_tensor(sidelength, path, channels=1):
    """
    Loads a video from a file, resizes each frame while maintaining aspect ratio,
    and normalizes the frame tensors.

    Args:
        sidelength (int): The target size for the longer side of the frames.
        path (str): The file path to the video.
        channels (int, optional): The number of channels (1 for greyscale, 3 for RGB).
    
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor of shape (T, C, H, W) for the video.
            - int: The new width of each frame.
            - int: The new height of each frame.
            - int: The total number of frames.
    """
    reader = imageio.get_reader(path)
    frames = []
    new_width, new_height = None, None

    for frame in reader:
        # imageio returns frames as numpy arrays in (H, W, C) format.
        # Convert the frame to a PIL image and adjust channels.
        if channels == 3:
            img = Image.fromarray(frame).convert("RGB")
        elif channels == 1:
            img = Image.fromarray(frame).convert("L")
        else:
            raise ValueError("Channels must be either 1 (greyscale) or 3 (RGB)")

        # For the first frame, compute the new size while keeping aspect ratio.
        if new_width is None or new_height is None:
            width, height = img.size
            if width >= height:
                new_width = sidelength
                new_height = int((height / width) * sidelength)
            else:
                new_height = sidelength
                new_width = int((width / height) * sidelength)

            # Define transforms for frames
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

        # Apply the transformation to each frame
        frame_tensor = transform(img)  # shape: (C, new_height, new_width)
        frames.append(frame_tensor)

    # Stack frames to form a video tensor: shape (T, C, H, W)
    video_tensor = torch.stack(frames, dim=0)
    num_frames = video_tensor.shape[0]
    return video_tensor, new_width, new_height, num_frames


class VideoDataset(Dataset):
    """
    Custom dataset for processing a video. This dataset loads all frames from a video, 
    resizes them, and generates a 3D grid of coordinates (time, height, width) along with the pixel values.
    """
    def __init__(self, sidelength, path, channels=1):
        """
        Initializes the VideoDataset.

        Args:
            sidelength (int): The target size for the longer side of the frames.
            path (str): The file path to the video.
            channels (int, optional): The number of channels (1 for greyscale, 3 for RGB).
        """
        super().__init__()
        # Load the video tensor, its frame dimensions, and the number of frames.
        video_tensor, self.width, self.height, self.num_frames = get_video_tensor(sidelength, path, channels)
        # video_tensor shape: (T, C, H, W)

        # Rearrange tensor to shape (T, H, W, C) and flatten all pixels across all frames.
        self.pixels = video_tensor.permute(0, 2, 3, 1).reshape(-1, channels)

        # Generate 3D grid coordinates for each pixel in every frame.
        self.coords = get_mgrid3d(self.num_frames, self.width, self.height)

    def __len__(self):
        # The dataset represents one whole video.
        return 1

    def __getitem__(self, idx):
        """
        Retrieves the 3D grid coordinates and pixel values for the video.

        Args:
            idx (int): Index of the item to retrieve. Only index 0 is valid since this dataset contains one video.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The grid coordinates of shape (T*H*W, 3), where each coordinate is (t, y, x).
                - torch.Tensor: The pixel values of shape (T*H*W, channels).
        """
        if idx > 0:
            raise IndexError("Index out of range. This dataset only contains one item (the whole video).")
        return self.coords, self.pixels
