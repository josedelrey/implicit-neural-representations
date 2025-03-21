import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from models import Siren
from loss import mse_to_psnr


def main():
    seed = 42

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parameters
    sidelength = 256
    is_rgb = True
    channels = 3 if is_rgb else 1
        
    image_dataset = ImageDataset(sidelength, path='images/test2.jpg', channels=channels)
    dataloader = DataLoader(image_dataset, batch_size=1, pin_memory=True, num_workers=0)

    model = Siren(in_features=2, out_features=channels, hidden_features=256, 
                      hidden_layers=3, outermost_linear=True).cuda()

    total_steps = 500
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output, coords = model(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

        optim.zero_grad()
        loss.backward()
        optim.step()

    # --- Rendering the final image ---
    # --- Rendering the final image ---
    model.eval()
    chunk_size = 4096  # Use chunks to avoid memory overflow
    with torch.no_grad():
        full_uv = image_dataset.coords.cuda()  # Use the same coordinate grid from the dataset
        preds_list = []
        for i in range(0, full_uv.shape[0], chunk_size):
            uv_chunk = full_uv[i:i+chunk_size]
            preds_chunk, _ = model(uv_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0)
        preds = preds.cpu().numpy()

    # Retrieve image dimensions from the dataset
    height, width = image_dataset.height, image_dataset.width

    if channels == 3:
        # For RGB images, reshape to (H, W, 3) and adjust pixel range to [0,1]
        final_img = preds.reshape(height, width, 3)
        final_img = (final_img + 1) / 2  # assuming the network output is in [-1, 1]
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img)
        plt.title("Reconstructed RGB Image by SIREN")
    else:
        # For grayscale, reshape to (H, W) and use a colormap
        final_img = preds.reshape(height, width)
        final_img = (final_img + 1) / 2  # adjust pixel range to [0,1] if needed
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img, cmap='gray')
        plt.title("Reconstructed Grayscale Image by SIREN")

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
