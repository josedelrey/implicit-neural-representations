import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.loss import mse_to_psnr
from models.model_factory import build_model


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Parameters
    task = 'image'
    image_path = 'images/cameraman.png'
    is_rgb = False
    sidelength = 256
    channels = 3 if is_rgb else 1
    total_steps = 1000
    log_interval = 10
    chunk_size = 4096
    model_type = 'waveletnet'

    # Data
    dataset  = ImageDataset(sidelength, path=image_path, channels=channels)
    loader   = DataLoader(dataset, batch_size=1, pin_memory=(device.type == 'cuda'), num_workers=0)
    height, width = dataset.height, dataset.width

    # Model and optimizer
    model, learning_rate = build_model(model_type, task, channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    coords, pixels = next(iter(loader))
    coords, pixels = coords.to(device), pixels.to(device)
    for step in range(total_steps + 1):
        coords_squeezed = coords.squeeze(0)
        preds = model(coords_squeezed)
        loss = ((preds - pixels) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(f"Step {step}, Loss {loss.item():.6f}, PSNR {mse_to_psnr(loss.item()):.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        full_uv = dataset.coords.to(device)
        predictions = []
        for i in range(0, full_uv.shape[0], chunk_size):
            predictions.append(model(full_uv[i:i+chunk_size]))
        preds_all = torch.cat(predictions, dim=0).cpu().numpy()

    # Visualization
    image = preds_all.reshape(height, width, channels) if channels==3 else preds_all.reshape(height, width)
    image = (image + 1) / 2
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap=None if channels==3 else 'gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
