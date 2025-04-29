import matplotlib.pyplot as plt
import numpy as np
import torch

from modules.dataset import VideoDataset
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
    task = 'video'
    video_path = 'videos/akiyo_cif.y4m'
    is_rgb = True
    sidelength = 256
    channels = 3 if is_rgb else 1
    total_steps = 10000
    log_interval = 10
    batch_size = 32768
    chunk_size = 1024
    model_type = 'experiment'

    # Data
    dataset = VideoDataset(sidelength, path=video_path, channels=channels)
    height, width, num_frames = dataset.height, dataset.width, dataset.num_frames
    coords = dataset.coords.to(device)
    pixels = dataset.pixels.to(device)
    num_coords = coords.shape[0]

    # Model and optimizer
    model, learning_rate = build_model(model_type, task, channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for step in range(total_steps + 1):
        optimizer.zero_grad()
        indices = torch.randint(0, num_coords, (batch_size,), device=coords.device)
        batch_coords = coords[indices]
        batch_pixels = pixels[indices]
        preds = model(batch_coords)
        loss = ((preds - batch_pixels) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(f"Step {step}, Loss {loss.item():.6f}, PSNR {mse_to_psnr(loss.item()):.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_list = []
        for i in range(0, num_coords, chunk_size):
            preds_list.append(model(coords[i:i+chunk_size]))
        preds_all = torch.cat(preds_list, dim=0).cpu().numpy()

    # Reshape to video format and compute average PSNR
    video_pred = preds_all.reshape(num_frames, height, width, channels)
    video_truth = dataset.pixels.cpu().numpy().reshape(num_frames, height, width, channels)
    psnr_vals = [mse_to_psnr(((video_pred[t] - video_truth[t])**2).mean()) for t in range(num_frames)]
    avg_psnr = np.mean(psnr_vals)
    print(f"Average PSNR over all frames: {avg_psnr:.6f}")

    # Visualization of first frame
    first = (video_pred[0] + 1) / 2
    if channels == 3:
        plt.figure(figsize=(6,6))
        plt.imshow(np.clip(first, 0, 1))
        plt.title("Reconstructed First Frame")
    else:
        plt.figure(figsize=(6,6))
        plt.imshow(np.clip(first[...,0], 0, 1), cmap='gray')
        plt.title("Reconstructed First Frame")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
