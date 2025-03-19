import os
import datetime
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import PixelDataset
from models import Siren
from loss import mse_to_psnr


def main():
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    
    # Load image
    image_path = 'images/test.jpg'
    image = imageio.imread(image_path)[..., :3] / 255.0
    H, W = image.shape[:2]
    
    # Create dataset and dataloader
    dataset = PixelDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Initialize model
    model = Siren()
    model.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # TensorBoard writer initialization
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"./logs/siren_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set number of iterations
    num_iterations = 150000
    
    # Training loop
    data_iter = iter(dataloader)
    for iteration in tqdm(range(num_iterations), desc="Training iterations"):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset the iterator if it reaches the end of the dataloader
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        uv = batch['uv'].to(device)
        rgb = batch['rgb'].to(device)
        
        optimizer.zero_grad()
        preds = model(uv)
        loss = loss_fn(preds, rgb)
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.6f}, PSNR: {mse_to_psnr(loss.item()):.2f} dB")
            writer.add_scalar('loss', loss.item(), iteration)
            writer.add_scalar('psnr', mse_to_psnr(loss.item()), iteration)
    
    # After training, reconstruct the full image using the trained model
    model.eval()
    chunk_size = 4096
    with torch.no_grad():
        full_uv = dataset.uv.to(device)
        preds_list = []
        for i in range(0, full_uv.shape[0], chunk_size):
            uv_chunk = full_uv[i:i+chunk_size]
            preds_chunk = model(uv_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0)
        preds = preds.cpu().numpy()
    
    # Reshape predictions to image dimensions
    pred_img = preds.reshape(H, W, 3)
    
    # Display the predicted image
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(pred_img, 0, 1))
    plt.title("Predicted Image by SIREN")
    plt.axis('off')
    plt.show()
    
    writer.close()

if __name__ == '__main__':
    main()
