import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.loss import mse_to_psnr
from models.model_factory import build_model


def main():
    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parameters
    sidelength = 256
    is_rgb = False
    channels = 3 if is_rgb else 1
    total_steps = 1000
    log_interval = 10
    chunk_size = 4096
    model_type = 'mfnwaveletnet'
    task = 'image'

    # Data
    dataset  = ImageDataset(sidelength, path='images/cameraman.png', channels=channels)
    loader   = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    height, width = dataset.height, dataset.width

    # Model & optimizer
    model, learning_rate = build_model(model_type, task, channels, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    coords, pixels = next(iter(loader))
    coords, pixels = coords.cuda(), pixels.cuda()
    for step in range(total_steps + 1):
        coords_squeezed = coords.squeeze(0)
        preds = model(coords_squeezed)
        loss = ((preds - pixels) ** 2).mean()

        if step % log_interval == 0:
            print(f"Step {step}, Loss {loss.item():.6f}, PSNR {mse_to_psnr(loss.item()):.6f}")

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if model_type == 'mfnwaveletnet':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        full_uv = dataset.coords.cuda()
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
