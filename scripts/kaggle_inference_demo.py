import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

# Import Generator and helper modules from the cloned repo
from model_architecture import Generator
from model_metrics import ssim
from quantize_model import fuse_generator_bn


def load_q7_weights(model, q7_txt_path, device):
    """
    Reads Q7 weights from text file, dequantizes them (/128.0) and loads into model.
    """
    print(f"[INFO] Reading Q7 weights from: {q7_txt_path}")
    with open(q7_txt_path, 'r') as f:
        q7_values = [float(line.strip()) for line in f if line.strip()]

    state_dict = model.state_dict()
    ptr = 0
    new_state_dict = {}

    for name, param in model.named_parameters():
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        if 'weight' in name or 'bias' in name:
            numel = param.numel()
            q7_slice = q7_values[ptr : ptr + numel]
            ptr += numel

            q7_tensor = torch.tensor(q7_slice, dtype=torch.float32, device=device)
            dequant_tensor = q7_tensor / 128.0
            new_state_dict[name] = dequant_tensor.view_as(param)

    model.load_state_dict(new_state_dict, strict=False)
    print("[INFO] Successfully loaded and dequantized Q7 weights into model.")
    return model


def calculate_psnr(img1, img2):
    """Calculates PSNR between two images in range [0, 1]"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Paths
    hr_image_path = "./00000001_000.png"
    weights_path = "./srgan_q7_weights_qat.txt"

    if not os.path.exists(hr_image_path):
        raise FileNotFoundError(f"Original image not found: {hr_image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # 1. Load Original HR Image (1024x1024, convert to Grayscale/RGB)
    print(f"[INFO] Loading Original HR Image: {hr_image_path}")
    hr_pil = Image.open(hr_image_path).convert('RGB')
    
    # 2. Generate LR Image (256x256) via Bicubic Downsampling (4x reduction)
    print("[INFO] Downsampling HR (1024x1024) -> LR (256x256) via Bicubic...")
    lr_transform = transforms.Resize((256, 256), interpolation=Image.BICUBIC)
    lr_pil = lr_transform(hr_pil)

    # 3. Generate Bicubic Upscaled baseline (1024x1024)
    print("[INFO] Generating Bicubic baseline upscaled (1024x1024)...")
    bicubic_transform = transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
    bicubic_pil = bicubic_transform(lr_pil)

    # 4. Initialize Generator and load Q7 weights
    print("[INFO] Initializing Generator model (4x upscale)...")
    base_model = Generator(upscale_factor=4).to(device)
    model = fuse_generator_bn(base_model)
    model = load_q7_weights(model, weights_path, device)
    model.eval()

    # 5. Run inference with SRGAN Q7 model
    print("[INFO] Running inference on LR image using SRGAN Q7 weights...")
    lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0.0, 1.0)
        sr_pil = to_pil_image(sr_tensor.squeeze(0).cpu())

    # 6. Save all 4 outputs
    print("[INFO] Saving output images...")
    hr_pil.save("1_original_hr.png")
    lr_pil.save("2_low_res_lr.png")
    bicubic_pil.save("3_upscaled_bicubic.png")
    sr_pil.save("4_reconstructed_srgan.png")
    print("[INFO] All 4 images saved successfully:")
    print("   - Original HR   : 1_original_hr.png")
    print("   - Low-Res LR    : 2_low_res_lr.png")
    print("   - Bicubic 4x    : 3_upscaled_bicubic.png")
    print("   - SRGAN Q7      : 4_reconstructed_srgan.png")

    # 7. Evaluate Metrics (PSNR & SSIM against Original HR)
    hr_np = np.array(hr_pil).astype(np.float32) / 255.0
    bicubic_np = np.array(bicubic_pil).astype(np.float32) / 255.0
    sr_np = np.array(sr_pil).astype(np.float32) / 255.0

    # Calculate PSNR
    psnr_bicubic = calculate_psnr(hr_np, bicubic_np)
    psnr_srgan = calculate_psnr(hr_np, sr_np)

    # Calculate SSIM (using pytorch metric logic)
    hr_t = to_tensor(hr_pil).unsqueeze(0).to(device)
    bicubic_t = to_tensor(bicubic_pil).unsqueeze(0).to(device)
    sr_t = to_tensor(sr_pil).unsqueeze(0).to(device)

    ssim_bicubic = ssim(bicubic_t, hr_t).item()
    ssim_srgan = ssim(sr_t, hr_t).item()

    print("\n" + "=" * 60)
    print("                 EVALUATION METRICS RESULTS                   ")
    print("=" * 60)
    print(f" Bicubic Baseline  : PSNR = {psnr_bicubic:.4f} dB | SSIM = {ssim_bicubic:.4f}")
    print(f" SRGAN Q7 (Ours)   : PSNR = {psnr_srgan:.4f} dB | SSIM = {ssim_srgan:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
