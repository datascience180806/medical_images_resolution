import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(cell_type, source, execution_count=None):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")[:-1]] + ([source.split("\n")[-1]] if source.split("\n")[-1] else [])
        }
        if cell_type == "code":
            cell["execution_count"] = execution_count
            cell["outputs"] = []
        nb["cells"].append(cell)

    # -------------------------------------------------------------
    # Cell 1: Markdown Title & Intro
    # -------------------------------------------------------------
    c1 = """# Comparative SISR Training Pipeline on Kaggle (Dual T4 GPUs)
## Models: SRCNN, ESPCN, FSRCNN, VDSR, EDSR on Medical X-Ray Dataset

This notebook provides an automated, multi-GPU training and evaluation pipeline for 5 classic Super-Resolution models on Kaggle:
1. **SRCNN** (*Dong et al., ECCV 2014*): 3-layer CNN with Bicubic pre-upsampling.
2. **ESPCN** (*Shi et al., CVPR 2016*): Post-upsampling with Sub-Pixel Convolution (PixelShuffle).
3. **FSRCNN** (*Dong et al., ECCV 2016*): Fast SRCNN with hourglass mapping and deconvolution.
4. **VDSR** (*Kim et al., CVPR 2016*): 20-layer Deep CNN with Global Residual Learning and gradient clipping.
5. **EDSR** (*Lim et al., CVPRW 2017*): Enhanced Deep Residual Network without BatchNorm.

### Kaggle Optimizations:
- **Dual T4 GPU Support**: Automatic `nn.DataParallel` distribution across 2x NVIDIA T4 GPUs.
- **Direct Kaggle Dataset Integration**: Directly reads input images mounted in `/kaggle/input/` without manual downloading.
- **Standalone Clean Weights**: Automatically strips `module.` prefixes so weights can be loaded anywhere (single GPU, CPU, or FPGA).
- **Preliminary PSNR & SSIM Evaluation**: Evaluates after each epoch and exports summary CSV and visual comparison plots.
- **One-Click ZIP Export**: Automatically archives all trained `.pth` models in `/kaggle/working/` for easy download."""
    add_cell("markdown", c1)

    # -------------------------------------------------------------
    # Cell 2: Imports & Kaggle Environment Setup
    # -------------------------------------------------------------
    c2 = """import os
import sys
import math
import time
import glob
import pickle
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor

# Detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle")
WORKING_DIR = "/kaggle/working" if IS_KAGGLE else "."
WEIGHTS_DIR = os.path.join(WORKING_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Multi-GPU Configuration
NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Running on Kaggle: {IS_KAGGLE}")
print(f"[INFO] Primary Device: {DEVICE}")
print(f"[INFO] Available GPUs: {NUM_GPUS}")
if NUM_GPUS > 0:
    for i in range(NUM_GPUS):
        print(f"       GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print(f"[INFO] Output Weights Directory: {WEIGHTS_DIR}")"""
    add_cell("code", c2)

    # -------------------------------------------------------------
    # Cell 3: Configuration & Hyperparameters
    # -------------------------------------------------------------
    c3 = """# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
UPSCALE_FACTOR = 2           # Scaling factor: 2, 3, or 4
CROP_SIZE = 128              # High-Resolution (HR) patch size
# With 2x T4 GPUs (32GB combined VRAM), batch size 32 or 64 is optimal
BATCH_SIZE = 32 if NUM_GPUS >= 2 else 16
NUM_EPOCHS = 20              # Number of epochs per model
LEARNING_RATE = 1e-4         # Learning rate for Adam optimizer
VAL_SPLIT_RATIO = 0.15       # 15% validation data
MAX_TRAIN_IMAGES = None      # Set to an integer (e.g. 10000) for faster runs, or None for full dataset

# List of models to train sequentially
MODELS_TO_TRAIN = ["SRCNN", "ESPCN", "FSRCNN", "VDSR", "EDSR"]

print(f"[CONFIG] Upscale Factor: {UPSCALE_FACTOR}x")
print(f"[CONFIG] Batch Size: {BATCH_SIZE} (Distributed across {max(1, NUM_GPUS)} GPU(s))")
print(f"[CONFIG] HR Crop: {CROP_SIZE}x{CROP_SIZE} | LR Input: {CROP_SIZE // UPSCALE_FACTOR}x{CROP_SIZE // UPSCALE_FACTOR}")
print(f"[CONFIG] Models queue: {MODELS_TO_TRAIN}")"""
    add_cell("code", c3)

    # -------------------------------------------------------------
    # Cell 4: Dataset Auto-Discovery for Kaggle & Local
    # -------------------------------------------------------------
    c4 = """# ==========================================
# DATASET DISCOVERY & PREPARATION
# ==========================================
def discover_dataset_images():
    \"\"\"
    Searches Kaggle /kaggle/input/ and local folders for medical X-ray images.
    Returns list of valid image file paths.
    \"\"\"
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    found_paths = []

    search_dirs = [
        "/kaggle/input",
        "./sub_X-Ray",
        "./data",
        "./eval_images"
    ]

    # 1. Check if Kaggle inputs exist
    if os.path.exists("/kaggle/input"):
        print("[INFO] Scanning /kaggle/input for mounted datasets...")
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if any(f.endswith(ext.replace("*", "")) for ext in extensions):
                    found_paths.append(os.path.join(root, f))
            if len(found_paths) >= 50000:  # Early stop if huge dataset
                break

    # 2. Check local splits (train_images.pkl) if applicable
    if len(found_paths) == 0 and os.path.exists("./data/train_images.pkl"):
        try:
            with open("./data/train_images.pkl", "rb") as fp:
                saved_paths = pickle.load(fp)
                valid_saved = [p for p in saved_paths if os.path.exists(p)]
                if len(valid_saved) > 0:
                    found_paths = valid_saved
                    print(f"[INFO] Loaded {len(found_paths)} images from ./data/train_images.pkl")
        except Exception as e:
            print(f"[DEBUG] Pickle read exception: {e}")

    # 3. Fallback scan on local directories
    if len(found_paths) == 0:
        for d in search_dirs:
            if os.path.isdir(d):
                for ext in extensions:
                    found_paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    found_paths = sorted(list(set(found_paths)))
    return found_paths

all_image_paths = discover_dataset_images()
print(f"[INFO] Total discovered image paths: {len(all_image_paths):,}")

if len(all_image_paths) == 0:
    raise FileNotFoundError(
        "No images found! Please ensure an X-Ray dataset is attached under Kaggle 'Data' "
        "or placed in './sub_X-Ray' or './data'."
    )

# Shuffle and split
random.seed(SEED)
random.shuffle(all_image_paths)

if MAX_TRAIN_IMAGES is not None and len(all_image_paths) > MAX_TRAIN_IMAGES:
    all_image_paths = all_image_paths[:MAX_TRAIN_IMAGES]
    print(f"[INFO] Dataset capped at {MAX_TRAIN_IMAGES:,} images for efficient training.")

val_count = max(1, int(len(all_image_paths) * VAL_SPLIT_RATIO))
train_image_paths = all_image_paths[val_count:]
val_image_paths = all_image_paths[:val_count]

print(f"[INFO] Training set: {len(train_image_paths):,} images")
print(f"[INFO] Validation set: {len(val_image_paths):,} images")


class MedicalSRDataset(Dataset):
    \"\"\"
    PyTorch Dataset generating paired (LR, HR) tensors.
    HR is randomly cropped and augmented, LR is generated via Bicubic downsampling.
    \"\"\"
    def __init__(self, image_paths, crop_size=128, upscale_factor=2, is_train=True):
        self.image_paths = image_paths
        self.crop_size = crop_size - (crop_size % upscale_factor)
        self.lr_size = self.crop_size // upscale_factor
        self.upscale_factor = upscale_factor
        self.is_train = is_train

        self.lr_downsample = transforms.Resize(
            (self.lr_size, self.lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            hr_img = img.convert("RGB")

        w, h = hr_img.size
        if w < self.crop_size or h < self.crop_size:
            hr_img = transforms.functional.resize(hr_img, (max(h, self.crop_size), max(w, self.crop_size)))
            w, h = hr_img.size

        if self.is_train:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            hr_crop = transforms.functional.crop(hr_img, top, left, self.crop_size, self.crop_size)

            if random.random() > 0.5:
                hr_crop = transforms.functional.hflip(hr_crop)
            if random.random() > 0.5:
                hr_crop = transforms.functional.vflip(hr_crop)
        else:
            hr_crop = transforms.functional.center_crop(hr_img, (self.crop_size, self.crop_size))

        lr_img = self.lr_downsample(hr_crop)

        hr_tensor = to_tensor(hr_crop)
        lr_tensor = to_tensor(lr_img)

        return lr_tensor, hr_tensor

# DataLoaders: Kaggle provides 4 vCPUs -> num_workers=4
num_workers = min(4, os.cpu_count() or 2)
train_dataset = MedicalSRDataset(train_image_paths, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, is_train=True)
val_dataset = MedicalSRDataset(val_image_paths, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, is_train=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

print(f"[INFO] DataLoader ready -> Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")"""
    add_cell("code", c4)

    # -------------------------------------------------------------
    # Cell 5: Evaluation Metrics (PSNR & SSIM)
    # -------------------------------------------------------------
    c5 = """# ==========================================
# EVALUATION METRICS: PSNR & SSIM
# ==========================================
def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    \"\"\"Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.\"\"\"
    mse = torch.mean((sr_tensor - hr_tensor) ** 2, dim=[1, 2, 3])
    zero_mask = (mse == 0)
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + 1e-10))
    psnr[zero_mask] = 100.0
    return psnr.mean().item()


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def calculate_ssim(img1, img2, window_size=11):
    \"\"\"Calculates Structural Similarity Index Measure (SSIM).\"\"\"
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()"""
    add_cell("code", c5)

    # -------------------------------------------------------------
    # Cell 6: Model Architectures
    # -------------------------------------------------------------
    c6 = """# ==========================================
# 5 SUPER-RESOLUTION MODEL ARCHITECTURES
# ==========================================

# 1. SRCNN (Dong et al., ECCV 2014)
class SRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


# 2. ESPCN (Shi et al., CVPR 2016)
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.tanh1(self.conv1(x))
        out = self.tanh2(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return torch.clamp(out, 0.0, 1.0)


# 3. FSRCNN (Dong et al., ECCV 2016)
class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(
            d, in_channels, kernel_size=9,
            stride=upscale_factor, padding=4,
            output_padding=upscale_factor - 1
        )

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)


# 4. VDSR (Kim et al., CVPR 2016)
class VDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, num_layers=20, num_features=64):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        self.residual_net = nn.Sequential(*layers)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        residual = self.residual_net(x_bicubic)
        out = x_bicubic + residual
        return torch.clamp(out, 0.0, 1.0)


# 5. EDSR (Lim et al., CVPRW 2017)
class EDSRResBlock(nn.Module):
    def __init__(self, n_feats=64, res_scale=0.1):
        super(EDSRResBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, n_feats=64, n_resblocks=8, res_scale=0.1):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[EDSRResBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        if upscale_factor in [2, 3]:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor)
            )
        elif upscale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f"Unsupported upscale factor: {upscale_factor}")

        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body_conv(self.body(x_head))
        res = res + x_head
        out = self.tail(self.upsampler(res))
        return torch.clamp(out, 0.0, 1.0)


def build_model(model_name, in_channels=3, upscale_factor=2):
    \"\"\"Factory function to instantiate models.\"\"\"
    name = model_name.upper()
    if name == "SRCNN":
        return SRCNN(in_channels=in_channels, upscale_factor=upscale_factor)
    elif name == "ESPCN":
        return ESPCN(in_channels=in_channels, upscale_factor=upscale_factor)
    elif name == "FSRCNN":
        return FSRCNN(in_channels=in_channels, upscale_factor=upscale_factor)
    elif name == "VDSR":
        return VDSR(in_channels=in_channels, upscale_factor=upscale_factor)
    elif name == "EDSR":
        return EDSR(in_channels=in_channels, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

print("===== MODEL PARAMETER COUNTS =====")
for m_name in ["SRCNN", "ESPCN", "FSRCNN", "VDSR", "EDSR"]:
    m = build_model(m_name, in_channels=3, upscale_factor=UPSCALE_FACTOR)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {m_name:<8} : {total_params:>10,d} parameters")"""
    add_cell("code", c6)

    # -------------------------------------------------------------
    # Cell 7: Multi-GPU Training & Validation Routines
    # -------------------------------------------------------------
    c7 = """# ==========================================
# TRAINING & VALIDATION ENGINES (DATA PARALLEL READY)
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=None):
    \"\"\"Executes 1 training epoch.\"\"\"
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for lr_imgs, hr_imgs in pbar:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        running_loss += loss.item() * lr_imgs.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    \"\"\"Evaluates model performance returning Loss, PSNR, and SSIM.\"\"\"
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for lr_imgs, hr_imgs in pbar:
        batch_size = lr_imgs.size(0)
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        running_loss += loss.item() * batch_size
        total_psnr += calculate_psnr(sr_imgs, hr_imgs) * batch_size
        total_ssim += calculate_ssim(sr_imgs, hr_imgs) * batch_size
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    return avg_loss, avg_psnr, avg_ssim"""
    add_cell("code", c7)

    # -------------------------------------------------------------
    # Cell 8: Main Multi-GPU Training Loop with Automatic Weight Export
    # -------------------------------------------------------------
    c8 = """# ==========================================================
# MAIN TRAINING PIPELINE: DUAL GPU SUPPORT & CHECKPOINT EXPORT
# ==========================================================
all_histories = {}
summary_records = []

criterion = nn.L1Loss().to(DEVICE)

for model_name in MODELS_TO_TRAIN:
    print(f"\\n{'=' * 65}")
    print(f"  STARTING TRAINING: {model_name} (Scale Factor: {UPSCALE_FACTOR}x)")
    print(f"{'=' * 65}")

    # Build base model
    base_model = build_model(model_name, in_channels=3, upscale_factor=UPSCALE_FACTOR).to(DEVICE)

    # Wrap with DataParallel if multiple GPUs (e.g. Kaggle 2x T4) are present
    if NUM_GPUS > 1:
        print(f"[INFO] Wrapping {model_name} with nn.DataParallel across {NUM_GPUS} GPUs...")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    clip_grad = 0.4 if model_name == "VDSR" else None

    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, clip_grad=clip_grad)
        val_loss, val_psnr, val_ssim = evaluate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch

            # Extract underlying clean state_dict without 'module.' prefix
            clean_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            # Export best model checkpoint
            best_weights_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}_best_{UPSCALE_FACTOR}x.pth")
            torch.save({
                "model_name": model_name,
                "upscale_factor": UPSCALE_FACTOR,
                "epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "state_dict": clean_state_dict
            }, best_weights_path)

        elapsed = time.time() - t_epoch_start
        best_marker = " [★ BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PSNR: {val_psnr:.4f} dB | SSIM: {val_ssim:.4f}{best_marker}")

    total_time_min = (time.time() - start_time) / 60.0
    print(f"[INFO] Completed {model_name} in {total_time_min:.2f} mins.")
    print(f"[INFO] Best Performance: PSNR = {best_psnr:.4f} dB, SSIM = {best_ssim:.4f} (at Epoch {best_epoch})")

    # Export latest weights checkpoint as pure state_dict
    clean_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    latest_weights_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}_latest_{UPSCALE_FACTOR}x.pth")
    torch.save(clean_state_dict, latest_weights_path)

    print(f"[EXPORT] Saved best checkpoint to: {best_weights_path}")
    print(f"[EXPORT] Saved latest state_dict to: {latest_weights_path}")

    # Record model summary
    weights_size_mb = os.path.getsize(best_weights_path) / (1024 * 1024)
    param_count = sum(p.numel() for p in (model.module if isinstance(model, nn.DataParallel) else model).parameters())

    summary_records.append({
        "Model": model_name,
        "Scale": f"{UPSCALE_FACTOR}x",
        "Parameters": f"{param_count:,}",
        "Best Epoch": best_epoch,
        "PSNR (dB)": round(best_psnr, 4),
        "SSIM": round(best_ssim, 4),
        "Weights File": os.path.basename(best_weights_path),
        "Size (MB)": f"{weights_size_mb:.2f} MB"
    })

    all_histories[model_name] = history

    # Clean cache between models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()"""
    add_cell("code", c8)

    # -------------------------------------------------------------
    # Cell 9: Benchmark Summary Table
    # -------------------------------------------------------------
    c9 = """# ==========================================
# PRELIMINARY BENCHMARK SUMMARY TABLE
# ==========================================
df_summary = pd.DataFrame(summary_records)
print(\"\\n=======================================================\")
print(\"  PRELIMINARY EVALUATION SUMMARY (PSNR & SSIM)\")
print(\"=======================================================\")
display(df_summary)

# Export summary to CSV file
csv_summary_path = os.path.join(WEIGHTS_DIR, f"benchmark_summary_{UPSCALE_FACTOR}x.csv")
df_summary.to_csv(csv_summary_path, index=False)
print(f\"\\n[INFO] Saved benchmark summary table to: {csv_summary_path}\")"""
    add_cell("code", c9)

    # -------------------------------------------------------------
    # Cell 10: Training Curves Visualization
    # -------------------------------------------------------------
    c10 = """# ==========================================
# TRAINING CURVES (LOSS, PSNR, SSIM)
# ==========================================
epochs_range = range(1, NUM_EPOCHS + 1)
colors = {"SRCNN": "#1f77b4", "ESPCN": "#ff7f0e", "FSRCNN": "#2ca02c", "VDSR": "#d62728", "EDSR": "#9467bd"}

plt.figure(figsize=(18, 5))

# Subplot 1: Validation Loss
plt.subplot(1, 3, 1)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_loss"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation Loss ({UPSCALE_FACTOR}x)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss (L1)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Subplot 2: PSNR (dB)
plt.subplot(1, 3, 2)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_psnr"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation PSNR ({UPSCALE_FACTOR}x)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Subplot 3: SSIM
plt.subplot(1, 3, 3)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_ssim"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation SSIM ({UPSCALE_FACTOR}x)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
curves_plot_path = os.path.join(WEIGHTS_DIR, f"training_curves_{UPSCALE_FACTOR}x.png")
plt.savefig(curves_plot_path, dpi=200)
plt.show()
print(f"[INFO] Saved training curves to: {curves_plot_path}")"""
    add_cell("code", c10)

    # -------------------------------------------------------------
    # Cell 11: Visual Super-Resolution Inference Comparison
    # -------------------------------------------------------------
    c11 = """# ==========================================================
# VISUAL INFERENCE: BICUBIC vs SRCNN vs ESPCN vs FSRCNN vs VDSR vs EDSR vs HR
# ==========================================================
sample_lr, sample_hr = val_dataset[0]
sample_lr_input = sample_lr.unsqueeze(0).to(DEVICE)
sample_hr_target = sample_hr.unsqueeze(0).to(DEVICE)

# Standard Bicubic interpolation baseline
bicubic_sr = F.interpolate(sample_lr_input, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)
bicubic_sr = torch.clamp(bicubic_sr, 0.0, 1.0)
bicubic_psnr = calculate_psnr(bicubic_sr, sample_hr_target)
bicubic_ssim = calculate_ssim(bicubic_sr, sample_hr_target)

reconstructions = {
    f"Bicubic Baseline\\n({bicubic_psnr:.2f} dB | {bicubic_ssim:.4f})": bicubic_sr.squeeze(0).cpu().permute(1, 2, 0).numpy()
}

# Run inference with each model using best exported weights
for m_name in MODELS_TO_TRAIN:
    weights_path = os.path.join(WEIGHTS_DIR, f"{m_name.lower()}_best_{UPSCALE_FACTOR}x.pth")
    if os.path.exists(weights_path):
        m = build_model(m_name, in_channels=3, upscale_factor=UPSCALE_FACTOR).to(DEVICE)
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        m.load_state_dict(state_dict)
        m.eval()

        with torch.no_grad():
            sr_tensor = m(sample_lr_input)
            psnr_val = calculate_psnr(sr_tensor, sample_hr_target)
            ssim_val = calculate_ssim(sr_tensor, sample_hr_target)
            sr_np = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            reconstructions[f"{m_name}\\n({psnr_val:.2f} dB | {ssim_val:.4f})"] = sr_np

# Append Ground Truth HR
reconstructions["Ground Truth (HR)\\n(Original Reference)"] = sample_hr.permute(1, 2, 0).numpy()

# Plot side-by-side reconstruction visual comparison
num_imgs = len(reconstructions)
fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4.5))

for ax, (title, img_arr) in zip(axes, reconstructions.items()):
    ax.imshow(np.clip(img_arr, 0.0, 1.0))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis("off")

plt.tight_layout()
visual_comp_path = os.path.join(WEIGHTS_DIR, f"visual_comparison_{UPSCALE_FACTOR}x.png")
plt.savefig(visual_comp_path, dpi=200)
plt.show()
print(f"[INFO] Saved visual comparison to: {visual_comp_path}")"""
    add_cell("code", c11)

    # -------------------------------------------------------------
    # Cell 12: Archive & Download Weights for Kaggle Users
    # -------------------------------------------------------------
    c12 = """# ==========================================================
# ARCHIVE & DOWNLOAD WEIGHTS (KAGGLE ONE-CLICK EXPORT)
# ==========================================================
zip_output_path = os.path.join(WORKING_DIR, f"sr_models_weights_{UPSCALE_FACTOR}x.zip")

print(f"[INFO] Packaging all weights from '{WEIGHTS_DIR}' into ZIP...")
with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(WEIGHTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, WEIGHTS_DIR)
            zipf.write(file_path, arcname)

zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
print(f"[SUCCESS] Archive ready: {zip_output_path} ({zip_size_mb:.2f} MB)")
print("👉 You can now download this ZIP file directly from the Kaggle Output tab in the right sidebar!")"""
    add_cell("code", c12)

    # Write notebook file
    target_path = "train_srcnn.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[SUCCESS] Successfully written {len(nb['cells'])} cells to {target_path}")

if __name__ == "__main__":
    create_notebook()
