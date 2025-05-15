import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os, sys
from cycler import cycler
from PIL import Image

sys.path.insert(0, os.getcwd())

from modules.dataset import ImageDataset
from modules.loss import mse_to_psnr
from models.mfn import WaveletNet


def main():
    # Parse command-line args
    parser = argparse.ArgumentParser(
        description="Train WaveletNet on all images in a folder for different omega0 values."
    )
    parser.add_argument("--image-dir", required=True,
                        help="Path to the folder containing input images.")
    parser.add_argument("--sidelength", type=int, default=256,
                        help="Spatial size of each input image.")
    parser.add_argument("--total-steps", type=int, default=1000,
                        help="Number of training steps per omega0.")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Steps between PSNR logging and printout.")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Chunk size for full-image evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--omegas", type=float, nargs='+',
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                        help="List of omega0 values to sweep.")
    parser.add_argument("--out-dir", default=".",
                        help="Directory to save the final comparison plot PDF into.")
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Gather image paths
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_paths = sorted([
        os.path.join(args.image_dir, fname)
        for fname in os.listdir(args.image_dir)
        if fname.lower().endswith(valid_exts)
    ])
    if not image_paths:
        print(f"No images found in {args.image_dir} with extensions {valid_exts}")
        sys.exit(1)

    omega0_list = args.omegas
    all_results = {}  # mapping from filename → list of max PSNRs

    # Process each image in the folder
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        # detect channels via PIL
        with Image.open(img_path) as im:
            mode = im.mode
        if mode == 'L':
            channels = 1
        else:
            channels = 3
        print(f"\n=== Processing image: {stem} (mode={mode}, channels={channels}) ===")

        # Prepare data loader
        dataset = ImageDataset(args.sidelength, path=img_path, channels=channels)
        loader = DataLoader(dataset, batch_size=1,
                            pin_memory=(device.type=='cuda'), num_workers=0)
        coords, pixels = next(iter(loader))
        coords = coords.to(device).squeeze(0)
        pixels = pixels.to(device)

        # Sweep omega0 and record best PSNR
        max_psnrs = []
        for omega0 in omega0_list:
            print(f"  ω₀ = {omega0:.2f}")
            model = WaveletNet(
                in_features=2, out_features=channels,
                hidden_layers=4, hidden_features=256,
                input_scale=128.0, weight_scale=1.0,
                alpha=6.0, beta=1.0,
                omega0=omega0, bias=True, output_act=False
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_psnr = 0.0

            model.train()
            for step in range(args.total_steps + 1):
                preds = model(coords)
                loss = ((preds - pixels) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % args.log_interval == 0:
                    psnr = mse_to_psnr(loss.item())
                    best_psnr = max(best_psnr, psnr)

            max_psnrs.append(best_psnr)
            print(f"    → max PSNR: {best_psnr:.4f} dB")

        all_results[stem] = max_psnrs

    # Plot comparison
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_prop_cycle(cycler('color', plt.get_cmap('Set2').colors))

    for stem, psnrs in all_results.items():
        plt.plot(
            omega0_list, psnrs,
            linestyle='-', linewidth=2,
            marker='o', markersize=6,
            markerfacecolor='none', markeredgewidth=1.5,
            label=stem
        )

    # Labels and title
    plt.xlabel("ω₀")
    plt.ylabel("Max PSNR (dB)")

    # Vertical and horizontal guide‐lines at each ω₀
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Force ticks at each integer ω₀
    plt.xticks(omega0_list)

    # Tidy up spines & ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)

    # Legend
    plt.legend(title="Image file", loc="best")

    # Save and show
    out_path = os.path.join(args.out_dir, "psnr_vs_omega0_comparison.pdf")
    plt.savefig(out_path, format='pdf')
    print(f"\nComparison plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
