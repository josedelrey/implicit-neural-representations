import imageio.v2 as imageio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pixel_dataset import PixelDataset

def main():
    image_path = 'test.jpg'
    image = imageio.imread(image_path)[..., :3] / 255.0  # shape: (H, W, 3)
    H, W = image.shape[:2]
    img_wh = (W, H)
    
    dataset = PixelDataset(image_path, img_wh)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create an empty image array to store the reconstructed image.
    reconstructed_img = np.zeros((H, W, 3))
    
    # Iterate over the DataLoader with a tqdm progress bar.
    for batch in tqdm(dataloader, desc="Processing batches"):
        uv_batch = batch['uv']   # shape: (batch_size, 2) with uv in [-1, 1]
        rgb_batch = batch['rgb']   # shape: (batch_size, 3)
        
        # Optionally, add a nested tqdm for each pixel in the batch:
        for i in range(uv_batch.shape[0]):
            uv = uv_batch[i].numpy()  
            rgb_val = rgb_batch[i].numpy()
            
            # Map uv from [-1, 1] to pixel indices.
            col = int(round((uv[0] + 1) * (W - 1) / 2))
            row = int(round((uv[1] + 1) * (H - 1) / 2))
            
            # Assign the rgb value to the reconstructed image.
            reconstructed_img[row, col, :] = rgb_val

    # Plot the reconstructed image.
    plt.figure(figsize=(6,6))
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed Image from Dataset")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
