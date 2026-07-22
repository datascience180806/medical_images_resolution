import os
import math
import argparse
import glob
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from model_architecture import Generator
from model_metrics import ssim


def is_image_file(filename):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return any(filename.endswith(ext) for ext in valid_extensions)


def evaluate_baseline(eval_dir, weights_path, output_csv, upscale_factor=4, device_str='auto'):
    """
    Evaluates the baseline Generator model (FP32) on a directory of evaluation images.
    
    Args:
        eval_dir (str): Path to directory containing evaluation HR images.
        weights_path (str): Path to Generator weights checkpoint (.pth.tar).
        output_csv (str): Path to output CSV file for metrics.
        upscale_factor (int): Upscale factor (default: 4).
        device_str (str): Device to use ('cuda', 'cpu', or 'auto').
    """
    # 1. Setup device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # 2. Load model
    print(f"[INFO] Initializing Generator (upscale_factor={upscale_factor})...")
    model = Generator(upscale_factor=upscale_factor).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    print(f"[INFO] Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. Collect evaluation images
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found at: {eval_dir}")

    image_paths = sorted([
        os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if is_image_file(f)
    ])
    print(f"[INFO] Found {len(image_paths)} evaluation images in '{eval_dir}'")

    if len(image_paths) == 0:
        print("[WARNING] No valid images found for evaluation.")
        return

    # Transforms for downsampling to LR (256x256)
    lr_transform = transforms.Resize((256, 256), interpolation=Image.BICUBIC)

    results = []
    total_psnr = 0.0
    total_ssim = 0.0

    print("[INFO] Starting evaluation...")
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating"):
            filename = os.path.basename(img_path)

            # Load HR image
            hr_pil = Image.open(img_path).convert('RGB')
            # Downsample to LR
            lr_pil = lr_transform(hr_pil)

            # Convert to tensors [1, C, H, W]
            hr_tensor = to_tensor(hr_pil).unsqueeze(0).to(device)
            lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)

            # Run inference
            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

            # Calculate MSE & PSNR
            mse_val = torch.mean((sr_tensor - hr_tensor) ** 2).item()
            if mse_val == 0:
                psnr_val = 100.0
            else:
                max_val = hr_tensor.max().item()
                psnr_val = 10.0 * math.log10((max_val ** 2) / mse_val)

            # Calculate SSIM
            ssim_val = ssim(sr_tensor, hr_tensor).item()

            total_psnr += psnr_val
            total_ssim += ssim_val

            results.append({
                "Filename": filename,
                "MSE": mse_val,
                "PSNR_dB": psnr_val,
                "SSIM": ssim_val
            })

    mean_psnr = total_psnr / len(image_paths)
    mean_ssim = total_ssim / len(image_paths)

    print("\n" + "=" * 50)
    print("           BASELINE EVALUATION RESULTS (FP32)       ")
    print("=" * 50)
    print(f" Total Images Evaluated : {len(image_paths)}")
    print(f" Mean PSNR              : {mean_psnr:.4f} dB")
    print(f" Mean SSIM              : {mean_ssim:.4f}")
    print("=" * 50 + "\n")

    # 4. Save results to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    
    # Append summary row at the bottom
    summary_df = pd.DataFrame([{
        "Filename": "AVERAGE_SUMMARY",
        "MSE": df["MSE"].mean(),
        "PSNR_dB": mean_psnr,
        "SSIM": mean_ssim
    }])
    df_final = pd.concat([df, summary_df], ignore_index=True)
    df_final.to_csv(output_csv, index=False)
    print(f"[INFO] Detailed evaluation metrics saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline FP32 Generator Model on Evaluation Dataset")
    parser.add_argument('--eval_dir', type=str, default='./eval_images', help='Path to directory containing HR evaluation images')
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to Generator checkpoint')
    parser.add_argument('--output_csv', type=str, default='./logs/baseline_fp32_metrics.csv', help='Output CSV path for metrics')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor (default: 4)')
    parser.add_argument('--device', type=str, default='auto', help="Device to run evaluation ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()
    evaluate_baseline(
        eval_dir=args.eval_dir,
        weights_path=args.weights,
        output_csv=args.output_csv,
        upscale_factor=args.upscale_factor,
        device_str=args.device
    )
