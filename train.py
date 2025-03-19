import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PixelDataset
from models import Siren
from loss import mse_to_psnr  # if needed

def main():
    # Set the device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load image and extract dimensions.
    image_path = 'images/test.jpg'
    image = imageio.imread(image_path)[..., :3] / 255.0  # shape: (H, W, 3)
    H, W = image.shape[:2]
    img_wh = (W, H)
    
    # Create dataset and dataloader.
    dataset = PixelDataset(image_path, img_wh)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Initialize the SIREN model and move it to the device.
    model = Siren(in_features=2, out_features=3,
                  hidden_features=256, hidden_layers=4,
                  outermost_linear=True,  # using a final linear layer for regression
                  first_omega_0=30, hidden_omega_0=30.)
    model.to(device)
    
    # Define loss function and optimizer.
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Set number of iterations.
    num_iterations = 5000  # Adjust the number of iterations as needed
    print("Starting training iterations...")
    
    # Create an iterator from the dataloader.
    data_iter = iter(dataloader)
    for iteration in tqdm(range(num_iterations), desc="Training iterations"):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset the iterator if we've reached the end.
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        uv = batch['uv'].to(device)   # shape: [batch_size, 2]
        rgb = batch['rgb'].to(device)   # shape: [batch_size, 3]
        
        optimizer.zero_grad()
        preds = model(uv)
        loss = loss_fn(preds, rgb)
        loss.backward()
        optimizer.step()
        
        # Optionally, print the loss every 100 iterations.
        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.6f}")
    
    # After training, reconstruct the full image using the trained model.
    model.eval()
    chunk_size = 1024  # Adjust the chunk size as needed to prevent memory overflow.
    with torch.no_grad():
        full_uv = dataset.uv.to(device)  # shape: (H*W, 2)
        preds_list = []
        # Process the full UV grid in chunks.
        for i in range(0, full_uv.shape[0], chunk_size):
            uv_chunk = full_uv[i:i+chunk_size]
            preds_chunk = model(uv_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0)
        preds = preds.cpu().numpy()  # shape: (H*W, 3)
    
    # Reshape predictions to image dimensions.
    pred_img = preds.reshape(H, W, 3)
    
    # Display the predicted image.
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(pred_img, 0, 1))
    plt.title("Predicted Image by SIREN")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
