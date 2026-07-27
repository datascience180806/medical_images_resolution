import os
import argparse
import math
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from model_architecture import Generator
from model_metrics import ssim
from quantize_model import fuse_generator_bn
from export_q7 import export_q7_weights


class STEQuantizeQ7(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Q7 Quantization.
    Forward pass: Simulates Q7 Quantization (Float32 -> Q7 -> Dequantized FP32)
    Backward pass: Passes gradients straight through without modifications.
    """
    @staticmethod
    def forward(ctx, input_tensor):
        scaled = input_tensor * 128.0
        rounded = torch.round(scaled)
        clamped = torch.clamp(rounded, -128, 127)
        return clamped / 128.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Q7QATConv2d(nn.Module):
    """
    Wrapper module replacing Conv2d during QAT training to simulate Q7 quantization.
    """
    def __init__(self, conv: nn.Conv2d):
        super(Q7QATConv2d, self).__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        
        self.weight = nn.Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Apply simulated Q7 quantization using STE
        q_w = STEQuantizeQ7.apply(self.weight)
        q_b = STEQuantizeQ7.apply(self.bias) if self.bias is not None else None
        
        return nn.functional.conv2d(
            x, q_w, q_b,
            stride=self.stride, padding=self.padding, groups=self.groups
        )


def replace_conv_with_qat(module):
    """
    Recursively replaces Conv2d layers with Q7QATConv2d.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, Q7QATConv2d(child))
        else:
            replace_conv_with_qat(child)


def prepare_qat_model(weights_path, upscale_factor=4, device=torch.device('cpu')):
    """
    Loads FP32 model, fuses BN, and prepares it for Quantization-Aware Training.
    """
    base_model = Generator(upscale_factor=upscale_factor).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        base_model.load_state_dict(checkpoint["model"])
    else:
        base_model.load_state_dict(checkpoint)
    base_model.eval()

    # 1. Fuse BN layers first (QAT must run on the hardware-fused model)
    fused_model = fuse_generator_bn(base_model)

    # 2. Quantize only the 16 Residual Blocks + Intermediate ConvBlock (FPGA PL Target)
    print("[INFO] Preparing 16 Residual Blocks and Intermediate ConvBlock for QAT...")
    for res_block in fused_model.residual:
        replace_conv_with_qat(res_block)
    replace_conv_with_qat(fused_model.convblock)

    return fused_model


class MicroQATDataset(Dataset):
    """
    Loads a micro-dataset from a directory containing images.
    Used for fast QAT fine-tuning on Kaggle without downloading the full 45GB dataset.
    """
    def __init__(self, image_dir):
        super(MicroQATDataset, self).__init__()
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_paths[index]).convert('RGB')
        return self.lr_transform(hr_image), self.hr_transform(hr_image)

    def __len__(self):
        return len(self.image_paths)


def train_qat(weights_path, train_dir, val_dir, output_dir, upscale_factor=4, epochs=15, batch_size=8, lr=1e-5, device_str='auto'):
    """
    QAT Fine-tuning loop.
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"[INFO] Running QAT on device: {device}")

    # 1. Clear GPU memory cache
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Prepare QAT Model
    model = prepare_qat_model(weights_path, upscale_factor, device).to(device)

    # 2. Data Loaders
    print(f"[INFO] Loading fine-tuning dataset from: {train_dir}")
    train_dataset = MicroQATDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(f"[INFO] Loading validation dataset from: {val_dir}")
    val_dataset = MicroQATDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 3. Optimizer & Criterion (QAT uses simple L1 or MSE loss to preserve original content)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_psnr = 0.0
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Starting QAT Fine-tuning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for lr_img, hr_img in train_bar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optimizer.zero_grad()
            sr_img = model(lr_img)
            
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lr_img.size(0)
            train_bar.set_description(f"QAT Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

        # Validation Pass
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for val_lr, val_hr in val_loader:
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)
                
                sr = model(val_lr)
                sr = torch.clamp(sr, 0.0, 1.0)
                
                mse_val = torch.mean((sr - val_hr) ** 2).item()
                psnr_val = 100.0 if mse_val == 0 else 10.0 * math.log10(1.0 / mse_val)
                ssim_val = ssim(sr, val_hr).item()

                total_psnr += psnr_val
                total_ssim += ssim_val

        mean_psnr = total_psnr / len(val_dataset)
        mean_ssim = total_ssim / len(val_dataset)

        print(f"[VAL] Epoch {epoch} - PSNR: {mean_psnr:.4f} dB | SSIM: {mean_ssim:.4f}")

        # Save checkpoint if accuracy improves
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_model_path = os.path.join(output_dir, "netG_4x_qat_best.pth")
            torch.save({"model": model.state_dict(), "qat": True}, best_model_path)
            print(f"[SAVE] New best model saved to {best_model_path} with PSNR: {best_psnr:.4f} dB")

    # 4. Export the final optimized Q7 weights to a txt file
    print("[EXPORT] Exporting best QAT weights to Q7 format...")
    best_weights_path = os.path.join(output_dir, "netG_4x_qat_best.pth")
    final_txt_output = os.path.join(output_dir, "srgan_q7_weights_qat.txt")
    export_q7_weights(best_weights_path, output_filename=final_txt_output, upscale_factor=upscale_factor, fuse_bn=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization-Aware Training (QAT) for Q7 FPGA weights")
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to FP32 weights')
    parser.add_argument('--train_dir', type=str, default='./calibration_images', help='Path to train/calibration folder')
    parser.add_argument('--val_dir', type=str, default='./eval_images', help='Path to eval folder')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save QAT output')
    parser.add_argument('--epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help="Device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()
    train_qat(
        weights_path=args.weights,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        upscale_factor=4,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device
    )
