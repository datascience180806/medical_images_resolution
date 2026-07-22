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
from quantize_model import fuse_generator_bn, replace_conv2d_with_quant


def is_image_file(filename):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return any(filename.endswith(ext) for ext in valid_extensions)


def load_model_from_checkpoint(weights_path, upscale_factor=4, device=torch.device('cpu')):
    """
    Loads FP32 or Quantized INT8 Generator model based on checkpoint content.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    is_quantized = False
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        if checkpoint.get("quantized") is True:
            is_quantized = True
            state_dict = checkpoint["model"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

    # Check if state_dict keys contain quantized layer markers
    if any("weight_int8" in key for key in state_dict.keys()):
        is_quantized = True

    base_model = Generator(upscale_factor=upscale_factor).to(device)

    if is_quantized:
        print("[INFO] Detected Quantized INT8 model structure. Preparing model layout...")
        fused_model = fuse_generator_bn(base_model)
        replace_conv2d_with_quant(fused_model)
        fused_model.to(device)
        
        # Turn off calibration on quantized layers for inference
        for m in fused_model.modules():
            if hasattr(m, "calibrating"):
                m.calibrating = False

        fused_model.load_state_dict(state_dict)
        model = fused_model
    else:
        print("[INFO] Loaded standard FP32 model structure.")
        base_model.load_state_dict(state_dict)
        model = base_model

    model.eval()
    return model, is_quantized


def evaluate_model(eval_dir, weights_path, output_csv, upscale_factor=4, device_str='auto'):
    """
    Evaluates Generator model (FP32 or Quantized INT8) on evaluation dataset.
    """
    # 1. Setup device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # 2. Load model
    print(f"[INFO] Loading model weights from: {weights_path}")
    model, is_quantized = load_model_from_checkpoint(weights_path, upscale_factor=upscale_factor, device=device)

    model_type_str = "INT8 Quantized" if is_quantized else "FP32 Baseline"

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

    print(f"[INFO] Starting evaluation ({model_type_str})...")
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"Evaluating [{model_type_str}]"):
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

    print("\n" + "=" * 55)
    print(f"           EVALUATION RESULTS ({model_type_str})       ")
    print("=" * 55)
    print(f" Model Type             : {model_type_str}")
    print(f" Total Images Evaluated : {len(image_paths)}")
    print(f" Mean PSNR              : {mean_psnr:.4f} dB")
    print(f" Mean SSIM              : {mean_ssim:.4f}")
    print("=" * 55 + "\n")

    # 4. Save results to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    
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
    parser = argparse.ArgumentParser(description="Evaluate Generator Model (FP32 or INT8 Quantized)")
    parser.add_argument('--eval_dir', type=str, default='./eval_images', help='Path to evaluation images')
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to checkpoint')
    parser.add_argument('--output_csv', type=str, default='./logs/evaluation_metrics.csv', help='Output CSV path')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--device', type=str, default='auto', help="Device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()
    evaluate_model(
        eval_dir=args.eval_dir,
        weights_path=args.weights,
        output_csv=args.output_csv,
        upscale_factor=args.upscale_factor,
        device_str=args.device
    )
