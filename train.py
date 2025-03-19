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

from dataset import ImageFitting
from models import Siren
from loss import mse_to_psnr  # assuming mse_to_psnr is defined in loss.py

def main():
    seed = 42  # or any integer of your choice

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                    hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output, coords = img_siren(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

        optim.zero_grad()
        loss.backward()
        optim.step()

    # --- Rendering the final image ---
    img_siren.eval()
    chunk_size = 4096  # Use chunks to avoid memory overflow
    with torch.no_grad():
        full_uv = cameraman.coords.cuda()  # Use the same coordinate grid from the dataset
        preds_list = []
        for i in range(0, full_uv.shape[0], chunk_size):
            uv_chunk = full_uv[i:i+chunk_size]
            preds_chunk, _ = img_siren(uv_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0)
        preds = preds.cpu().numpy()

    # Reshape predictions to image dimensions (since it's grayscale, shape will be H x W)
    final_img = preds.reshape(256, 256)

    plt.figure(figsize=(6, 6))
    plt.imshow(final_img, cmap='gray')
    plt.title("Reconstructed Image by SIREN")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
