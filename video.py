import matplotlib.pyplot as plt
import numpy as np
import torch

# Use VideoDataset instead of ImageDataset:
from modules.dataset import VideoDataset
from modules.mlp import MLP
from modules.siren import Siren
from modules.mfn import FourierNet, GaborNet, MFNWaveletNet
from modules.wavelet import WaveletNet
from modules.wire import WIRE
from modules.finer import Finer
from modules.frinr import FRINR
from modules.loss import mse_to_psnr

def main():
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parameters
    sidelength = 256
    is_rgb = True
    channels = 3 if is_rgb else 1
    total_steps = 20000
    log_interval = 10
    batch_size_train = 32768     # Batch size for random mini-batch sampling in training
    chunk_size_eval = 1024       # Chunk size used during evaluation
    model_type = 'gabornet'      # Options: 'mlp', 'siren', 'gabornet', etc.

    # Load the video dataset directly (bypassing the dataloader since the dataset returns one item)
    video_dataset = VideoDataset(sidelength, path='videos/akiyo_cif.y4m', channels=channels)
    # Get frame dimensions and total number of frames
    height, width, num_frames = video_dataset.height, video_dataset.width, video_dataset.num_frames

    # Instead of using the dataloader, directly retrieve the full video tensors:
    # video_dataset.coords has shape (T*H*W, 3)
    # video_dataset.pixels has shape (T*H*W, channels)
    all_coords = video_dataset.coords.cuda()
    all_pixels = video_dataset.pixels.cuda()
    num_coords = all_coords.shape[0]

    # Define the number of input features.
    # For video, the coordinates are (t, y, x) so input dimension is 3.
    in_features = 3

    # Initialize the model based on model_type.
    if model_type == 'mlp':
        model = MLP(n_in=in_features,
                    n_out=channels,
                    n_layers=5,
                    n_hidden_units=1024,
                    act='gaussian',
                    act_trainable=True,
                    use_pe=False,
                    L=10,
                    a=0.1).cuda()
        learning_rate = 1e-3

    elif model_type == 'siren':
        model = Siren(in_features=in_features,
                      hidden_features=1024,
                      hidden_layers=4,
                      out_features=channels,
                      outermost_linear=True,
                      first_omega_0=30,
                      hidden_omega_0=30).cuda()
        learning_rate = 1e-4

    elif model_type == 'gabornet':
        model = GaborNet(in_size=in_features,
                         hidden_size=1024,
                         out_size=channels,
                         n_layers=3,
                         input_scale=256.0,
                         weight_scale=1.0,
                         alpha=6.0,
                         beta=1.0,
                         bias=True,
                         output_act=False).cuda()
        learning_rate = 1e-3

    elif model_type == 'fouriernet':
        model = FourierNet(in_size=in_features,
                           hidden_size=1024,
                           out_size=channels,
                           n_layers=3,
                           input_scale=256.0,
                           weight_scale=1.0,
                           bias=True,
                           output_act=False).cuda()
        learning_rate = 1e-3

    elif model_type == 'mfnwaveletnet':
        model = MFNWaveletNet(in_size=in_features,
                              hidden_size=1024,
                              out_size=channels,
                              n_layers=3,
                              input_scale=128.0,
                              weight_scale=1.0,
                              alpha=6.0,
                              beta=1.0,
                              omega0=5.0).cuda()
        learning_rate = 1e-3

    elif model_type == 'waveletnet':
        model = WaveletNet(in_features=in_features,
                           hidden_features=1024,
                           out_features=channels,
                           hidden_layers=4,
                           omega0=5.0).cuda()
        learning_rate = 1e-3

    elif model_type == 'wire':
        model = WIRE(in_features=in_features,
                     hidden_features=1024,
                     hidden_layers=3,
                     out_features=channels,
                     outermost_linear=True,
                     first_omega_0=10.0,
                     hidden_omega_0=10.0,
                     scale=6.0,
                     pos_encode=False,
                     L=6).cuda()
        learning_rate = 1e-3

    elif model_type == 'finer':
        model = Finer(in_features=in_features,
                      out_features=channels,
                      hidden_layers=3,
                      hidden_features=1024,
                      first_omega=30,
                      hidden_omega=30,
                      init_method='sine',
                      init_gain=1,
                      fbs=None,
                      hbs=None,
                      alphaType=None,
                      alphaReqGrad=False).cuda()
        learning_rate = 1e-4

    elif model_type == 'frinr':
        model = FRINR(mode='sin', in_features=in_features,
                      hidden_features=1024,
                      hidden_layers=3,
                      out_features=channels,
                      outermost_linear=True,
                      high_freq_num=128,
                      low_freq_num=128,
                      phi_num=32,
                      alpha=0.01,  # for relu, alpha:0.05; for sin, alpha:0.01
                      first_omega_0=30.0,
                      hidden_omega_0=30.0,
                      pe=False).cuda()
        learning_rate = 1e-4

    # Initialize optimizer
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Training loop using random mini-batch sampling.
    for step in range(total_steps + 1):
        optim.zero_grad()
        # Sample a random batch of indices
        indices = torch.randint(0, num_coords, (batch_size_train,), device=all_coords.device)
        input_batch = all_coords[indices]
        truth_batch = all_pixels[indices]
        
        # Forward pass on the random mini-batch
        output_batch = model(input_batch)
        loss = ((output_batch - truth_batch) ** 2).mean()

        # Backpropagation and optimizer step
        loss.backward()
        optim.step()

        if step % log_interval == 0:
            psnr_val = mse_to_psnr(loss.cpu().item())
            print("Step %d, Loss: %0.6f, PSNR: %0.6f" % (step, loss.item(), psnr_val))

    # Evaluation: reconstruct the video in chunks and compute average pSNR over all frames.
    model.eval()
    with torch.no_grad():
        full_coords = video_dataset.coords.cuda()  # shape: (T*H*W, 3)
        preds_list = []
        for i in range(0, full_coords.shape[0], chunk_size_eval):
            coord_chunk = full_coords[i:i+chunk_size_eval]
            preds_chunk = model(coord_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0).cpu()

    # Reshape predictions and ground truth pixels into video format: (num_frames, height, width, channels)
    preds_video = preds.reshape(num_frames, height, width, channels)
    truth_video = video_dataset.pixels.cpu().reshape(num_frames, height, width, channels)

    # Convert from normalized range [-1, 1] to [0, 1]
    preds_video = (preds_video + 1) / 2
    truth_video = (truth_video + 1) / 2

    # Compute per-frame pSNR and average over all frames.
    psnr_values = []
    for t in range(num_frames):
        mse = ((preds_video[t] - truth_video[t]) ** 2).mean().item()
        psnr_values.append(mse_to_psnr(mse))
    avg_psnr = np.mean(psnr_values)
    print("Average pSNR over all frames: %0.6f" % avg_psnr)
    
    # Optionally, visualize the first reconstructed frame.
    if channels == 3:
        first_frame = preds_video[0]
        plt.figure(figsize=(6, 6))
        plt.imshow(first_frame)
        plt.title("Reconstructed First Frame")
    else:
        first_frame = preds_video[0].squeeze(-1)
        plt.figure(figsize=(6, 6))
        plt.imshow(first_frame, cmap='gray')
        plt.title("Reconstructed First Frame")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
