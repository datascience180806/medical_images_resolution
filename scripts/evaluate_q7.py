import os
import math
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from model_architecture import Generator
from model_metrics import ssim
from quantize_model import fuse_generator_bn


def load_q7_weights_to_model(model, q7_txt_path, device):
    """
    Reads the flattened Q7 integers from the txt file, dequantizes them (divides by 128.0),
    and reconstructs the model parameters to run inference.
    """
    if not os.path.exists(q7_txt_path):
        raise FileNotFoundError(f"Q7 weight file not found: {q7_txt_path}")

    print(f"[INFO] Reading Q7 weights from: {q7_txt_path}")
    with open(q7_txt_path, 'r') as f:
        q7_values = [float(line.strip()) for line in f if line.strip()]

    print(f"[INFO] Total Q7 parameters read: {len(q7_values)}")

    state_dict = model.state_dict()
    ptr = 0

    # We update the state dict parameters sequentially as exported
    new_state_dict = {}
    for name, param in model.named_parameters():
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
            
        if 'weight' in name or 'bias' in name:
            numel = param.numel()
            # Slice the flattened array and convert back to Float32
            q7_slice = q7_values[ptr : ptr + numel]
            ptr += numel

            # Dequantize: float32_val = q7_val / 128.0
            q7_tensor = torch.tensor(q7_slice, dtype=torch.float32, device=device)
            dequant_tensor = q7_tensor / 128.0
            
            # Reshape back to the original parameter shape
            new_state_dict[name] = dequant_tensor.view_as(param)

    # Load reconstructed state dict back to model
    model.load_state_dict(new_state_dict, strict=False)
    print("[INFO] Successfully loaded reconstructed Q7 weights into model.")
    return model


def evaluate_q7_model(eval_dir, q7_weights_path, output_csv, upscale_factor=4, device_str='auto'):
    """
    Inference evaluation workflow using Q7 dequantized weights.
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # Initialize model architecture and fuse BN
    base_model = Generator(upscale_factor=upscale_factor).to(device)
    model = fuse_generator_bn(base_model)

    # Load and dequantize Q7 weights
    model = load_q7_weights_to_model(model, q7_weights_path, device)
    model.eval()

    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    image_paths = sorted([
        os.path.join(eval_dir, f) for f in os.listdir(eval_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))
    ])
    print(f"[INFO] Found {len(image_paths)} images for evaluation.")

    lr_transform = transforms.Resize((256, 256), interpolation=Image.BICUBIC)

    results = []
    total_psnr = 0.0
    total_ssim = 0.0

    print("[INFO] Evaluating model quality (Q7)...")
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating Q7"):
            filename = os.path.basename(img_path)

            hr_pil = Image.open(img_path).convert('RGB')
            lr_pil = lr_transform(hr_pil)

            hr_tensor = to_tensor(hr_pil).unsqueeze(0).to(device)
            lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)

            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

            mse_val = torch.mean((sr_tensor - hr_tensor) ** 2).item()
            psnr_val = 100.0 if mse_val == 0 else 10.0 * math.log10(1.0 / mse_val)
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
    print("           EVALUATION RESULTS (Q7 Quantized)       ")
    print("=" * 55)
    print(f" Mean PSNR              : {mean_psnr:.4f} dB")
    print(f" Mean SSIM              : {mean_ssim:.4f}")
    print("=" * 55 + "\n")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Detailed Q7 metrics saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dequantized Q7 Model on Test Dataset")
    parser.add_argument('--eval_dir', type=str, default='./eval_images', help='Path to eval images')
    parser.add_argument('--q7_weights', type=str, default='./models/srgan_q7_weights.txt', help='Path to Q7 weights text file')
    parser.add_argument('--output_csv', type=str, default='./logs/q7_evaluation_metrics.csv', help='Output CSV metrics path')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--device', type=str, default='auto', help="Device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()
    evaluate_q7_model(
        eval_dir=args.eval_dir,
        q7_weights_path=args.q7_weights,
        output_csv=args.output_csv,
        upscale_factor=args.upscale_factor,
        device_str=args.device
    )
