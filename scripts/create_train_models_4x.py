import json
import os

def create_4x_notebook():
    with open('train_models_3x.ipynb', 'r', encoding='utf-8') as f:
        nb3 = json.load(f)

    nb4 = {
        "nbformat": nb3.get("nbformat", 4),
        "nbformat_minor": nb3.get("nbformat_minor", 5),
        "metadata": nb3.get("metadata", {}),
        "cells": []
    }

    # Cell 0: Markdown
    cell0_text = """# Comparative SISR Training Pipeline on Kaggle (Dual T4 GPUs) — Scale 4x (10 Epochs)
## Models: SRCNN, ESPCN, FSRCNN, VDSR, EDSR, SRGAN on Medical X-Ray Dataset

This notebook provides an automated, multi-GPU training and evaluation pipeline for **6 classic Super-Resolution models at 4x scale factor**, optimized for execution on Kaggle environment equipped with **2x NVIDIA Tesla T4 GPUs** (or single GPU / CPU fallback).

### Key Features & Adaptations for 4x Scale:
1. **Target Upscale Factor**: `4x` ($128 \\times 128$ HR patch reconstructed from $32 \\times 32$ LR input).
2. **Models Benchmarked**:
   - **SRCNN**: Baseline Pre-upsampling CNN ($9\\text{-}1\\text{-}5$ structure, bicubic pre-interpolated).
   - **ESPCN**: Efficient Sub-Pixel Convolutional Network ($48$ channels expanded to $3 \\times 4^2$ with `PixelShuffle(4)`).
   - **FSRCNN**: Fast Super-Resolution CNN with Deconvolution layer (`ConvTranspose2d` stride=4, padding=4, output_padding=3).
   - **VDSR**: 20-layer Deep Residual Network with global residual learning.
   - **EDSR**: Enhanced Deep Residual Network with 8 residual blocks and dual cascaded $2\\times$ PixelShuffle stages (`Lim et al., CVPRW 2017`).
   - **SRGAN**: Canonical 16-residual-block Generator with dual cascaded $2\\times$ PixelShuffle stages (`Ledig et al., CVPR 2017`).
3. **Dataset Sampling & Leakage Prevention**:
   - Discovers NIH Chest X-ray dataset across 12 subfolders (`images_001` ... `images_012`), sampling exactly 1,000 images per batch (total 12,000 images).
   - Automatically excludes any benchmark evaluation images in `eval_images` to prevent data leakage.
4. **Export & Packaging**:
   - Best checkpoints (`{model}_best_4x.pth`) and latest state dicts (`{model}_latest_4x.pth`).
   - Automated evaluation curves (`training_curves_4x.png`), visual inference grid (`visual_comparison_4x.png`), summary table (`benchmark_summary_4x.csv`), and one-click ZIP packaging (`sr_models_weights_4x.zip`).
"""
    nb4["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in cell0_text.splitlines()]
    })

    # Cell 1: Environment & Setup
    cell1_text = """import os
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
WEIGHTS_DIR = os.path.join(WORKING_DIR, "weights_4x")
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
print(f"[INFO] Output Weights Directory: {WEIGHTS_DIR}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell1_text.splitlines()]
    })

    # Cell 2: Hyperparameters & Configuration
    cell2_text = """# ==========================================
# HYPERPARAMETERS & CONFIGURATION (SCALE 4X)
# ==========================================
UPSCALE_FACTOR = 4           # Scaling factor: 4x
CROP_SIZE = 128              # High-Resolution (HR) patch size (128 / 4 = 32 clean integer)
BATCH_SIZE = 32 if NUM_GPUS >= 2 else 16
NUM_EPOCHS = 10              # 10 epochs for efficient benchmark training
LEARNING_RATE = 1e-4         # Learning rate for Adam optimizer
VAL_SPLIT_RATIO = 0.15       # 15% validation data
MAX_TRAIN_IMAGES = None      # Set to integer (e.g. 10000) or None for full dataset

# List of models to train sequentially
MODELS_TO_TRAIN = ["SRCNN", "ESPCN", "FSRCNN", "VDSR", "EDSR", "SRGAN"]

print(f"[CONFIG] Upscale Factor: {UPSCALE_FACTOR}x")
print(f"[CONFIG] Epochs per Model: {NUM_EPOCHS}")
print(f"[CONFIG] Batch Size: {BATCH_SIZE} (Distributed across {max(1, NUM_GPUS)} GPU(s))")
print(f"[CONFIG] HR Crop: {CROP_SIZE}x{CROP_SIZE} | LR Input: {CROP_SIZE // UPSCALE_FACTOR}x{CROP_SIZE // UPSCALE_FACTOR}")
print(f"[CONFIG] Models Queue: {MODELS_TO_TRAIN}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell2_text.splitlines()]
    })

    # Cell 3: Dataset Discovery & Preparation
    cell3_text = """# ==========================================
# DATASET DISCOVERY & PREPARATION
# ==========================================
def discover_dataset_images():
    \"\"\"
    Searches Kaggle /kaggle/input/ and local folders for medical X-ray images.
    Returns list of valid image file paths.
    \"\"\"
    valid_exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    found_paths = []

    # 1. Benchmark images to exclude (anti-leakage)
    eval_names = set()
    for eval_dir in ["./eval_images", "../eval_images", "/kaggle/working/eval_images"]:
        if os.path.exists(eval_dir):
            eval_names.update({os.path.basename(p) for p in glob.glob(os.path.join(eval_dir, "*.*"))})

    candidate_roots = [
        "/kaggle/input",
        "./X-Ray images/NIH",
        "./X-Ray images",
        "./sub_X-Ray",
        "./data"
    ]

    # 2. Fast Direct Path Match for Kaggle NIH Dataset
    # (e.g. /kaggle/input/datasets/nih-chest-xrays/data/images_001/images/)
    detected_folders = []
    direct_bases = [
        "/kaggle/input/datasets/nih-chest-xrays/data",
        "/kaggle/input/nih-chest-xrays/data",
        "/kaggle/input/data"
    ]
    for base in direct_bases:
        if os.path.isdir(base):
            candidate_list = []
            for i in range(1, 13):
                p1 = os.path.join(base, f"images_{i:03d}", "images")
                p2 = os.path.join(base, f"images_{i:03d}")
                if os.path.isdir(p1):
                    candidate_list.append(p1)
                elif os.path.isdir(p2):
                    candidate_list.append(p2)
            if len(candidate_list) >= 2:
                detected_folders = candidate_list
                print(f"[INFO] Fast-path detected NIH dataset at: {base}")
                break

    # 3. Fallback: Search for directories matching images_001 ... images_012
    if len(detected_folders) == 0:
        for c_root in candidate_roots:
            if os.path.isdir(c_root):
                for root, dirs, _ in os.walk(c_root):
                    for d in dirs:
                        if re.search(r"images?[-_]?0*(?:[1-9]|1[0-2])$", d, re.IGNORECASE):
                            full_d = os.path.join(root, d)
                            sub_img = os.path.join(full_d, "images")
                            target_dir = sub_img if os.path.isdir(sub_img) else full_d
                            if target_dir not in detected_folders:
                                detected_folders.append(target_dir)

    detected_folders = sorted(detected_folders)

    # 4. Sample up to 1000 from each folder
    if len(detected_folders) >= 2:
        print(f"[INFO] Detected {len(detected_folders)} NIH batch subfolders:")
        for idx, f_path in enumerate(detected_folders[:12], 1):
            folder_imgs = []
            for root, _, files in os.walk(f_path):
                for f in files:
                    if f.endswith(valid_exts) and f not in eval_names:
                        folder_imgs.append(os.path.join(root, f))
            folder_imgs.sort()
            sampled = folder_imgs[:1000]
            found_paths.extend(sampled)
            folder_name = os.path.basename(os.path.dirname(f_path) if f_path.endswith('images') else f_path)
            print(f"       [{idx:02d}/12] {folder_name}: {len(folder_imgs):,} available -> Sampled {len(sampled):,}")
    else:
        print("[INFO] NIH 12 subfolders not detected as separate directories. Performing recursive scan...")
        raw_imgs = []
        for c_root in candidate_roots:
            if os.path.isdir(c_root):
                for root, _, files in os.walk(c_root):
                    for f in files:
                        if f.endswith(valid_exts) and f not in eval_names:
                            raw_imgs.append(os.path.join(root, f))
                    if len(raw_imgs) >= 50000:
                        break
        raw_imgs = sorted(list(set(raw_imgs)))
        found_paths = raw_imgs[:12000]
        print(f"[INFO] Global fallback collected {len(found_paths):,} images.")

    if len(eval_names) > 0:
        print(f"[SECURITY] Successfully excluded {len(eval_names)} evaluation benchmark images (eval_images) to prevent data leakage.")
    print(f"[INFO] Total discovered dataset images: {len(found_paths):,}")
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
    PyTorch Dataset generating paired (LR, HR) tensors for 4x super-resolution.
    HR is randomly cropped (128x128) and augmented, LR is generated via Bicubic downsampling (32x32).
    \"\"\"
    def __init__(self, image_paths, crop_size=128, upscale_factor=4, is_train=True):
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

# Safe worker count for Kaggle environment
num_workers = min(2, os.cpu_count() or 2)
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

print(f"[INFO] DataLoader ready -> Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell3_text.splitlines()]
    })

    # Cell 4: Evaluation Metrics
    cell4_text = """# ==========================================
# EVALUATION METRICS: PSNR & SSIM
# ==========================================
def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    \"\"\"Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.\"\"\"
    mse = torch.mean((sr_tensor - hr_tensor) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr.mean().item()


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_ssim_window(window_size=11, channel=3):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def calculate_ssim(img1, img2, window_size=11, channel=3):
    \"\"\"Calculates Structural Similarity Index (SSIM).\"\"\"
    window = create_ssim_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()

print("[INFO] Evaluation metric functions (PSNR, SSIM) initialized successfully.")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell4_text.splitlines()]
    })

    # Cell 5: Model Architectures for 4x
    cell5_text = """# ==========================================
# MODEL ARCHITECTURES (4X CONFIGURATION)
# ==========================================

# 1. SRCNN (Dong et al., ECCV 2014) - Pre-upsampling 9-1-5 Architecture
class SRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # Pre-upsampling to target HR grid via bicubic interpolation
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


# 2. ESPCN (Shi et al., CVPR 2016) - Post-upsampling Sub-Pixel Conv
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()
        # For 4x: in_channels * (4^2) = 3 * 16 = 48 channels
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.tanh1(self.conv1(x))
        out = self.tanh2(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return torch.clamp(out, 0.0, 1.0)


# 3. FSRCNN (Dong et al., ECCV 2016) - Post-upsampling Deconvolution
class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, d=56, s=12, m=4):
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
        # Deconvolution for 4x: stride=4, padding=4, output_padding=3
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


# 4. VDSR (Kim et al., CVPR 2016) - 20-layer Deep Residual Network
class VDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, num_layers=20, num_features=64):
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


# 5. EDSR (Lim et al., CVPRW 2017) - Enhanced Deep Residual Network
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
    def __init__(self, in_channels=3, upscale_factor=4, n_feats=64, n_resblocks=8, res_scale=0.1):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[EDSRResBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # For 4x: canonical architecture uses two cascaded 2x PixelShuffle stages (Lim et al.)
        if upscale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor)
            )
        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body_conv(self.body(x_head))
        res = res + x_head
        out = self.tail(self.upsampler(res))
        return torch.clamp(out, 0.0, 1.0)


# 6. SRGAN Generator (Canonical Ledig et al., CVPR 2017)
class SRGANResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(SRGANResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class SRGAN(nn.Module):
    \"\"\"Canonical SRGAN Generator with 16 Residual Blocks for 4x upsampling.\"\"\"
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(SRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=num_channels)
        )
        self.residual = nn.Sequential(*[SRGANResidualBlock(num_channels) for _ in range(num_blocks)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        if upscale_factor == 4:
            # Canonical 4x upsampler: 2 consecutive 2x PixelShuffle stages (Ledig et al., CVPR 2017)
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(num_parameters=num_channels),
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(num_parameters=num_channels)
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.PReLU(num_parameters=num_channels)
            )
        self.final_conv = nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        res = self.residual(initial)
        mid = self.mid_conv(res) + initial
        up = self.upsampler(mid)
        out = (torch.tanh(self.final_conv(up)) + 1.0) / 2.0
        return torch.clamp(out, 0.0, 1.0)


def build_model(model_name, in_channels=3, upscale_factor=4):
    \"\"\"Factory function to instantiate models for scale 4x.\"\"\"
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
    elif name == "SRGAN":
        return SRGAN(in_channels=in_channels, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

print("===== 4X MODEL PARAMETER COUNTS =====")
for m_name in ["SRCNN", "ESPCN", "FSRCNN", "VDSR", "EDSR", "SRGAN"]:
    m = build_model(m_name, in_channels=3, upscale_factor=UPSCALE_FACTOR)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {m_name:<8} : {total_params:>10,d} parameters")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell5_text.splitlines()]
    })

    # Cell 6: Training & Validation Engines
    cell6_text = """# ==========================================
# TRAINING & VALIDATION ENGINES (DATA PARALLEL READY)
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=None):
    \"\"\"Executes 1 training epoch.\"\"\"
    model.train()
    running_loss = 0.0

    for lr_batch, hr_batch in dataloader:
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        sr_batch = model(lr_batch)
        loss = criterion(sr_batch, hr_batch)

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        running_loss += loss.item() * lr_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_epoch(model, dataloader, criterion, device):
    \"\"\"Evaluates model performance across validation set.\"\"\"
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for lr_batch, hr_batch in dataloader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)

            b_size = lr_batch.size(0)
            running_loss += loss.item() * b_size
            total_psnr += calculate_psnr(sr_batch, hr_batch) * b_size
            total_ssim += calculate_ssim(sr_batch, hr_batch) * b_size
            num_samples += b_size

    val_loss = running_loss / num_samples
    val_psnr = total_psnr / num_samples
    val_ssim = total_ssim / num_samples
    return val_loss, val_psnr, val_ssim

print("[INFO] Training and evaluation engines initialized.")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell6_text.splitlines()]
    })

    # Cell 7: Main Training Loop
    cell7_text = """# ==========================================================
# MAIN TRAINING PIPELINE: DUAL GPU SUPPORT & CHECKPOINT EXPORT
# ==========================================================
all_histories = {}
summary_records = []

criterion = nn.L1Loss().to(DEVICE)

for model_name in MODELS_TO_TRAIN:
    print(f"\\n{'=' * 65}")
    print(f"  STARTING TRAINING: {model_name} (Scale Factor: {UPSCALE_FACTOR}x | Epochs: {NUM_EPOCHS})")
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

    # Clip gradients for VDSR, FSRCNN and SRGAN for training stability
    clip_grad = 0.4 if model_name in ["VDSR", "FSRCNN", "SRGAN"] else None

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
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell7_text.splitlines()]
    })

    # Cell 8: Preliminary Benchmark Summary Table
    cell8_text = """# ==========================================
# PRELIMINARY BENCHMARK SUMMARY TABLE (4X)
# ==========================================
df_summary = pd.DataFrame(summary_records)
print("\\n=======================================================")
print("  PRELIMINARY EVALUATION SUMMARY (PSNR & SSIM - SCALE 4X)")
print("=======================================================")
display(df_summary)

# Export summary to CSV file
csv_summary_path = os.path.join(WEIGHTS_DIR, f"benchmark_summary_{UPSCALE_FACTOR}x.csv")
df_summary.to_csv(csv_summary_path, index=False)
print(f"\\n[INFO] Saved benchmark summary table to: {csv_summary_path}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell8_text.splitlines()]
    })

    # Cell 9: Training Curves
    cell9_text = """# ==========================================
# TRAINING CURVES (LOSS, PSNR, SSIM - SCALE 4X)
# ==========================================
epochs_range = range(1, NUM_EPOCHS + 1)
colors = {
    "SRCNN": "#1f77b4",
    "ESPCN": "#ff7f0e",
    "FSRCNN": "#2ca02c",
    "VDSR": "#d62728",
    "EDSR": "#9467bd",
    "SRGAN": "#8c564b"
}

plt.figure(figsize=(18, 5))

# Subplot 1: Validation Loss
plt.subplot(1, 3, 1)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_loss"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation Loss ({UPSCALE_FACTOR}x - 10 Epochs)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss (L1)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Subplot 2: PSNR (dB)
plt.subplot(1, 3, 2)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_psnr"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation PSNR ({UPSCALE_FACTOR}x - 10 Epochs)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Subplot 3: SSIM
plt.subplot(1, 3, 3)
for m_name, hist in all_histories.items():
    plt.plot(epochs_range, hist["val_ssim"], label=m_name, color=colors.get(m_name), linewidth=2)
plt.title(f"Validation SSIM ({UPSCALE_FACTOR}x - 10 Epochs)", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
curves_plot_path = os.path.join(WEIGHTS_DIR, f"training_curves_{UPSCALE_FACTOR}x.png")
plt.savefig(curves_plot_path, dpi=200)
plt.show()
print(f"[INFO] Saved training curves to: {curves_plot_path}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell9_text.splitlines()]
    })

    # Cell 10: Visual Comparison Grid
    cell10_text = """# ==========================================================
# VISUAL INFERENCE: BICUBIC vs SR MODELS vs HR (4X)
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
print(f"[INFO] Saved visual comparison to: {visual_comp_path}")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell10_text.splitlines()]
    })

    # Cell 11: Archive & Download 4x Weights
    cell11_text = """# ==========================================================
# ARCHIVE & DOWNLOAD 4X WEIGHTS (KAGGLE ONE-CLICK EXPORT)
# ==========================================================
zip_output_path = os.path.join(WORKING_DIR, f"sr_models_weights_{UPSCALE_FACTOR}x.zip")

print(f"[INFO] Packaging all 4x weights from '{WEIGHTS_DIR}' into ZIP...")
with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(WEIGHTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, WEIGHTS_DIR)
            zipf.write(file_path, arcname)

zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
print(f"[SUCCESS] 4x Weights Archive ready: {zip_output_path} ({zip_size_mb:.2f} MB)")
print(f"[ACTION REQUIRED] In Kaggle right-sidebar 'Output', expand '/kaggle/working' and click Download on 'sr_models_weights_4x.zip'!")
"""
    nb4["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell11_text.splitlines()]
    })

    with open('train_models_4x.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb4, f, indent=1, ensure_ascii=False)
    print("Generated train_models_4x.ipynb successfully!")

if __name__ == '__main__':
    create_4x_notebook()
