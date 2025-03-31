import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from models import Siren, GaborNet, FourierNet, WaveletNet, ReLUPE, MLP
from loss import mse_to_psnr


def main():
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parameters
    sidelength = 256
    is_rgb = False
    channels = 3 if is_rgb else 1
    total_steps = 5000
    log_interval = 10
    chunk_size = 4096
    model_type = 'waveletnet'

    # Load the image dataset
    image_dataset = ImageDataset(sidelength, path='images/cameraman.png', channels=channels)
    dataloader = DataLoader(image_dataset, batch_size=1, pin_memory=True, num_workers=0)

    # Retrieve image dimensions from the dataset
    height, width = image_dataset.height, image_dataset.width

    # Initialize model
    if model_type == 'mlp':
        model = MLP(n_in=2, 
                    n_out=channels, 
                    n_layers=5, 
                    n_hidden_units=256, 
                    act='relu', 
                    act_trainable=False, 
                    use_pe=True, 
                    L=10).cuda()
        learning_rate = 5e-4
    elif model_type == 'siren':
        model = Siren(in_features=2,
                      hidden_features=256,
                      hidden_layers=4,
                      out_features=channels,
                      outermost_linear=True,
                      first_omega_0=30,
                      hidden_omega_0=30).cuda()
        learning_rate = 1e-4
    elif model_type == 'gabornet':
        model = GaborNet(in_size=2,
                         hidden_size=256,
                         out_size=channels,
                         n_layers=4,
                         input_scale=256.0,
                         weight_scale=1.0,
                         alpha=6.0,
                         beta=1.0,
                         bias=True,
                         output_act=False).cuda()
        learning_rate = 1e-2
    elif model_type == 'fouriernet':
        model = FourierNet(in_size=2,
                           hidden_size=256,
                           out_size=channels,
                           n_layers=5,
                           input_scale=256.0,
                           weight_scale=1.0,
                           bias=True,
                           output_act=False).cuda()
        learning_rate = 1e-2
    elif model_type == 'waveletnet':
        model = WaveletNet(out_features=channels, hidden_layers=4).cuda()
        learning_rate = 1e-3

    # Initialize optimizer
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Training loop
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    for step in range(total_steps + 1):
        model_input = model_input.squeeze(0)
        model_output = model(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        
        if not step % log_interval:
            print("Step %d, Total loss: %0.6f, PSNR: %0.6f" % (step, loss, mse_to_psnr(loss.cpu().item())))

        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        full_uv = image_dataset.coords.cuda()
        preds_list = []
        for i in range(0, full_uv.shape[0], chunk_size):
            uv_chunk = full_uv[i:i+chunk_size]
            preds_chunk = model(uv_chunk)
            preds_list.append(preds_chunk)
        preds = torch.cat(preds_list, dim=0)
        preds = preds.cpu().numpy()

    if channels == 3:
        final_img = preds.reshape(height, width, 3)
        final_img = (final_img + 1) / 2
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img)
        plt.title("Reconstructed Image")
    else:
        final_img = preds.reshape(height, width)
        final_img = (final_img + 1) / 2
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img, cmap='gray')
        plt.title("Reconstructed Image")

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
