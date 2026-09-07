import json
import os

def create_2x_notebook():
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
    # Cell 1: Markdown Overview
    # -------------------------------------------------------------
    c1 = """# 🩺 Medical Image Super-Resolution Benchmark (Scale 2x)
## 6-Model Comparative Suite: SRCNN, ESPCN, FSRCNN, VDSR, EDSR, and SRGAN
### NIH ChestX-ray14 (12 Folders × 1,000 Images = 12,000 Images) with Strict Anti-Data-Leakage

---

### Key Features of this Pipeline:
1. **Target Upscale Factor:** `2x` magnification ($64 \\times 64 \\rightarrow 128 \\times 128$).
2. **NIH 12-Folder Smart Ingestion:** Samples exactly 1,000 images from each of the 12 NIH batch folders (`images_001` ... `images_012`) for a total of 12,000 images, preventing Kaggle session timeout while maintaining broad demographic representation.
3. **Strict Data-Leakage Prevention:** Automatically detects and excludes all evaluation benchmark images (`eval_images`) so the test set remains completely unseen.
4. **6 Evaluated Models:**
   - **SRCNN** (Dong et al., ECCV 2014) - Pre-upsampling baseline (hardware streaming friendly)
   - **ESPCN** (Shi et al., CVPR 2016) - Efficient sub-pixel convolution (PixelShuffle)
   - **FSRCNN** (Dong et al., ECCV 2016) - Post-upsampling deconvolution
   - **VDSR** (Kim et al., CVPR 2016) - Deep residual pre-upsampling
   - **EDSR** (Lim et al., CVPRW 2017) - Enhanced deep residual network
   - **SRGAN** (Ledig et al., CVPR 2017 / Swift-SRGAN) - Separable Conv Residual Generator
5. **Training Length:** 10 Epochs per model with gradient stabilization and safe PyTorch DataLoader configuration."""
    add_cell("markdown", c1)

    # -------------------------------------------------------------
    # Cell 2: Imports & Environment Setup
    # -------------------------------------------------------------
    c2 = """import os
import re
import glob
import math
import time
import random
import zipfile
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Global Seeds & Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detect Execution Environment
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = "/kaggle/working" if IS_KAGGLE else "."
WEIGHTS_DIR = os.path.join(WORKING_DIR, "weights_2x")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# GPU & Hardware Diagnostics
NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Platform: {'Kaggle Dual/Single GPU' if IS_KAGGLE else 'Local/Dedicated Machine'}")
print(f"[INFO] Primary Device: {DEVICE}")
print(f"[INFO] Total GPUs Available: {NUM_GPUS}")
if NUM_GPUS > 0:
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"       GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print(f"[INFO] Target Checkpoints Directory: {WEIGHTS_DIR}")"""
    add_cell("code", c2)

    # -------------------------------------------------------------
    # Cell 3: Configuration & Hyperparameters (Scale 2x)
    # -------------------------------------------------------------
    c3 = """# ==========================================================
# HYPERPARAMETERS & CONFIGURATION (SCALE 2X)
# ==========================================================
UPSCALE_FACTOR = 2            # Scaling factor: 2x
CROP_SIZE = 128               # HR patch size (128x128 -> LR is 64x64)
LR_SIZE = CROP_SIZE // UPSCALE_FACTOR
BATCH_SIZE = 32 if NUM_GPUS >= 2 else 16
NUM_EPOCHS = 10               # 10 epochs per model for efficient execution
LEARNING_RATE = 1e-4          # Base learning rate (Adam optimizer)
VAL_SPLIT_RATIO = 0.15        # 15% validation split
IMAGES_PER_NIH_FOLDER = 1000  # 1,000 images per folder x 12 folders = 12,000 images

# 6 Models to train sequentially for fair comparative benchmarking
MODELS_TO_TRAIN = ["SRCNN", "ESPCN", "FSRCNN", "VDSR", "EDSR", "SRGAN"]

print(f"[CONFIG] Upscale Factor: {UPSCALE_FACTOR}x")
print(f"[CONFIG] HR Crop: {CROP_SIZE}x{CROP_SIZE} | LR Input: {LR_SIZE}x{LR_SIZE}")
print(f"[CONFIG] Batch Size: {BATCH_SIZE} (Distributed over {max(1, NUM_GPUS)} GPU(s))")
print(f"[CONFIG] Epochs per Model: {NUM_EPOCHS}")
print(f"[CONFIG] Target Sampling: 12 folders × {IMAGES_PER_NIH_FOLDER} = {12 * IMAGES_PER_NIH_FOLDER:,} images")
print(f"[CONFIG] Models Queue ({len(MODELS_TO_TRAIN)}): {MODELS_TO_TRAIN}")"""
    add_cell("code", c3)

    # -------------------------------------------------------------
    # Cell 4: NIH 12-Folder Dataset Discovery with Anti-Leakage
    # -------------------------------------------------------------
    c4 = """# ==========================================================
# NIH 12-FOLDER DATASET DISCOVERY & ANTI-DATA-LEAKAGE
# ==========================================================
def discover_nih_12folders(max_per_folder=1000):
    \"\"\"
    Scans for NIH ChestX-ray14 batch directories (images_001 to images_012)
    and samples exactly `max_per_folder` images from each folder.
    Strictly excludes any benchmark images from `./eval_images` to prevent data leakage.
    \"\"\"
    valid_exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    found_paths = []

    # 1. Identify benchmark images to exclude
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

    # 3. Sample up to max_per_folder from each folder
    if len(detected_folders) >= 2:
        print(f"[INFO] Detected {len(detected_folders)} NIH batch subfolders:")
        for idx, f_path in enumerate(detected_folders[:12], 1):
            folder_imgs = []
            for root, _, files in os.walk(f_path):
                for f in files:
                    if f.endswith(valid_exts) and f not in eval_names:
                        folder_imgs.append(os.path.join(root, f))
            folder_imgs.sort()
            sampled = folder_imgs[:max_per_folder]
            found_paths.extend(sampled)
            folder_name = os.path.basename(os.path.dirname(f_path) if f_path.endswith('images') else f_path)
            print(f"       [{idx:02d}/12] {folder_name}: {len(folder_imgs):,} available -> Sampled {len(sampled):,}")
    else:
        # Fallback: scan candidate directories
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
        found_paths = raw_imgs[: 12 * max_per_folder]
        print(f"[INFO] Global fallback collected {len(found_paths):,} images.")

    if len(eval_names) > 0:
        print(f"[SECURITY] Successfully excluded {len(eval_names)} evaluation benchmark images (eval_images) to prevent data leakage.")
    print(f"[INFO] Total discovered dataset images: {len(found_paths):,}")
    return found_paths

all_image_paths = discover_nih_12folders(max_per_folder=IMAGES_PER_NIH_FOLDER)

if len(all_image_paths) == 0:
    raise FileNotFoundError(
        "No images found! Please ensure NIH ChestX-ray dataset is attached under Kaggle 'Data' "
        "or placed in './X-Ray images' or './sub_X-Ray'."
    )

# Shuffle and split into Train / Validation
random.seed(SEED)
random.shuffle(all_image_paths)

val_count = max(1, int(len(all_image_paths) * VAL_SPLIT_RATIO))
train_image_paths = all_image_paths[val_count:]
val_image_paths = all_image_paths[:val_count]

print(f"[INFO] Training set:   {len(train_image_paths):,} images")
print(f"[INFO] Validation set: {len(val_image_paths):,} images")"""
    add_cell("code", c4)

    # -------------------------------------------------------------
    # Cell 5: PyTorch Dataset & DataLoader
    # -------------------------------------------------------------
    c5 = """# ==========================================================
# PYTORCH DATASET & DATALOADER (SCALE 2X)
# ==========================================================
class MedicalSRDataset(Dataset):
    \"\"\"
    PyTorch Dataset generating paired (LR, HR) tensors for 2x super-resolution.
    HR is randomly cropped (128x128) and augmented; LR is bicubic downsampled (64x64).
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

            # Data Augmentation (Flips)
            if random.random() > 0.5:
                hr_crop = transforms.functional.hflip(hr_crop)
            if random.random() > 0.5:
                hr_crop = transforms.functional.vflip(hr_crop)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            hr_crop = transforms.functional.crop(hr_img, top, left, self.crop_size, self.crop_size)

        hr_tensor = transforms.functional.to_tensor(hr_crop)
        lr_tensor = self.lr_downsample(hr_tensor)

        return lr_tensor, hr_tensor

# Instantiate DataLoaders (safe num_workers=2 to prevent Kaggle multiprocessing crash)
train_dataset = MedicalSRDataset(train_image_paths, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, is_train=True)
val_dataset = MedicalSRDataset(val_image_paths, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, is_train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"[INFO] Train batches per epoch: {len(train_loader)} (Batch size: {BATCH_SIZE})")
print(f"[INFO] Val batches per epoch:   {len(val_loader)}")"""
    add_cell("code", c5)

    # -------------------------------------------------------------
    # Cell 6: 6 Model Architectures (SRCNN, ESPCN, FSRCNN, VDSR, EDSR, SRGAN)
    # -------------------------------------------------------------
    c6 = """# ==========================================================
# 6 MODEL ARCHITECTURES DEFINITION (SCALE 2X)
# ==========================================================

# 1. SRCNN (Dong et al., ECCV 2014) - Pre-upsampling baseline
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
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_up))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


# 2. ESPCN (Shi et al., CVPR 2016) - Post-upsampling Sub-Pixel Conv
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2):
        super(ESPCN, self).__init__()
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


# 3. FSRCNN (Dong et al., ECCV 2016) - Post-upsampling Deconvolution
class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
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
        # Deconvolution for 2x upscaling
        self.deconv = nn.ConvTranspose2d(d, in_channels, kernel_size=9, stride=upscale_factor, padding=4, output_padding=upscale_factor - 1)

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)


# 4. VDSR (Kim et al., CVPR 2016) - 20-layer Deep Residual Network
class ConvReLUBlock(nn.Module):
    def __init__(self):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, num_layers=20):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.residual_layers = nn.Sequential(*[ConvReLUBlock() for _ in range(num_layers - 2)])
        self.conv_last = nn.Conv2d(64, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        residual = self.conv_first(x_up)
        residual = self.residual_layers(residual)
        residual = self.conv_last(residual)
        out = torch.add(x_up, residual)
        return torch.clamp(out, 0.0, 1.0)


# 5. EDSR (Lim et al., CVPRW 2017) - Enhanced Deep Residual Network
class ResBlock(nn.Module):
    def __init__(self, channels=64, res_scale=0.1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, num_channels=64, num_blocks=8):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.head(x)
        b = self.body_conv(self.body(h)) + h
        out = self.tail(b)
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
    \"\"\"
    Canonical SRGAN Generator (Ledig et al., CVPR 2017).
    16 Residual Blocks with standard Conv2d and PixelShuffle upsampling.
    100% robust and memory-aligned across multi-GPU CUDA/cuDNN.
    \"\"\"
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=2):
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
        num_upsample = max(1, int(math.log2(upscale_factor)))
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers.extend([
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(num_parameters=num_channels)
            ])
        self.upsampler = nn.Sequential(*upsample_layers)
        self.final_conv = nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        res = self.residual(initial)
        mid = self.mid_conv(res) + initial
        up = self.upsampler(mid)
        out = (torch.tanh(self.final_conv(up)) + 1.0) / 2.0
        return torch.clamp(out, 0.0, 1.0)


# Model Factory Builder
def build_model(model_name, upscale_factor=2):
    model_map = {
        "SRCNN": lambda: SRCNN(in_channels=3, upscale_factor=upscale_factor),
        "ESPCN": lambda: ESPCN(in_channels=3, upscale_factor=upscale_factor),
        "FSRCNN": lambda: FSRCNN(in_channels=3, upscale_factor=upscale_factor),
        "VDSR": lambda: VDSR(in_channels=3, upscale_factor=upscale_factor),
        "EDSR": lambda: EDSR(in_channels=3, upscale_factor=upscale_factor),
        "SRGAN": lambda: SRGAN(in_channels=3, upscale_factor=upscale_factor)
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    return model_map[model_name]()

print("[INFO] Model Factory initialized successfully for all 6 models.")
for m_name in MODELS_TO_TRAIN:
    m = build_model(m_name, upscale_factor=UPSCALE_FACTOR)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"       {m_name:<7}: {params:>9,} trainable parameters")"""
    add_cell("code", c6)

    # -------------------------------------------------------------
    # Cell 7: Metrics & Loss Functions
    # -------------------------------------------------------------
    c7 = """# ==========================================================
# METRIC CALCULATION (PSNR & SSIM)
# ==========================================================
def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    \"\"\"Calculates PSNR (dB) between SR and HR tensors [B, C, H, W].\"\"\"
    mse = torch.mean((sr_tensor - hr_tensor) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse.item())

def calculate_ssim(img1, img2, window_size=11):
    \"\"\"Calculates SSIM index between two batches of normalized tensors [B, C, H, W].\"\"\"
    channel = img1.size(1)
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()"""
    add_cell("code", c7)

    # -------------------------------------------------------------
    # Cell 8: Unified Training & Evaluation Function
    # -------------------------------------------------------------
    c8 = """# ==========================================================
# UNIFIED TRAINING & VALIDATION ENGINE (SCALE 2X)
# ==========================================================
def train_single_model(model_name, upscale_factor=2, num_epochs=10, lr=1e-4):
    print(\"\\n\" + \"=\" * 65)
    print(f\"   STARTING TRAINING: {model_name} (Scale {upscale_factor}x | {num_epochs} Epochs)\")
    print(\"=\" * 65)

    model = build_model(model_name, upscale_factor=upscale_factor)
    model = model.to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)

    # L1 Loss for sharp edges, gradient clipping for deep models
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = 0
    history = {\"epoch\": [], \"train_loss\": [], \"val_loss\": [], \"val_psnr\": [], \"val_ssim\": []}
    start_time = time.time()

    best_ckpt_path = os.path.join(WEIGHTS_DIR, f\"{model_name.lower()}_best_2x.pth\")
    latest_ckpt_path = os.path.join(WEIGHTS_DIR, f\"{model_name.lower()}_latest_2x.pth\")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(DEVICE, non_blocking=True)
            hr_imgs = hr_imgs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()

            # Gradient clipping for VDSR, FSRCNN, SRGAN stability
            if model_name in [\"VDSR\", \"FSRCNN\", \"SRGAN\"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)

            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        scheduler.step()

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(DEVICE, non_blocking=True)
                hr_imgs = hr_imgs.to(DEVICE, non_blocking=True)

                sr_imgs = model(lr_imgs)
                val_loss = criterion(sr_imgs, hr_imgs)
                running_val_loss += val_loss.item()

                psnr = calculate_psnr(sr_imgs, hr_imgs)
                ssim = calculate_ssim(sr_imgs, hr_imgs)
                val_psnr_list.append(psnr)
                val_ssim_list.append(ssim)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_psnr = float(np.mean(val_psnr_list))
        avg_val_ssim = float(np.mean(val_ssim_list))
        epoch_duration = time.time() - epoch_start

        history[\"epoch\"].append(epoch)
        history[\"train_loss\"].append(avg_train_loss)
        history[\"val_loss\"].append(avg_val_loss)
        history[\"val_psnr\"].append(avg_val_psnr)
        history[\"val_ssim\"].append(avg_val_ssim)

        # Checkpoint Saving
        unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint_dict = {
            \"model_name\": model_name,
            \"upscale_factor\": upscale_factor,
            \"epoch\": epoch,
            \"best_epoch\": best_epoch,
            \"best_psnr\": best_psnr,
            \"best_ssim\": best_ssim,
            \"state_dict\": unwrapped_model.state_dict(),
            \"history\": history
        }

        # Save Latest Checkpoint
        torch.save(checkpoint_dict, latest_ckpt_path)

        # Save Best Checkpoint
        is_best = avg_val_psnr > best_psnr
        if is_best:
            best_psnr = avg_val_psnr
            best_ssim = avg_val_ssim
            best_epoch = epoch
            checkpoint_dict[\"best_epoch\"] = best_epoch
            checkpoint_dict[\"best_psnr\"] = best_psnr
            checkpoint_dict[\"best_ssim\"] = best_ssim
            torch.save(checkpoint_dict, best_ckpt_path)

        print(f\"[{model_name}] Epoch {epoch:02d}/{num_epochs:02d} ({epoch_duration:.1f}s) | \"
              f\"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | \"
              f\"PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f} \"
              f\"{'🌟 [NEW BEST]' if is_best else ''}\")

    total_training_time = time.time() - start_time
    print(f\"\\n[COMPLETED] {model_name} (2x) in {total_training_time/60:.2f} mins. \"
          f\"Best PSNR: {best_psnr:.2f} dB, Best SSIM: {best_ssim:.4f} at Epoch {best_epoch}.\")

    return {
        \"model_name\": model_name,
        \"best_epoch\": best_epoch,
        \"best_psnr\": best_psnr,
        \"best_ssim\": best_ssim,
        \"training_time_min\": total_training_time / 60,
        \"best_weights_path\": best_ckpt_path,
        \"history\": history
    }"""
    add_cell("code", c8)

    # -------------------------------------------------------------
    # Cell 9: Sequential Execution Loop
    # -------------------------------------------------------------
    c9 = """# ==========================================================
# EXECUTE SEQUENTIAL TRAINING FOR ALL 6 MODELS (SCALE 2X)
# ==========================================================
benchmark_results = []

for model_name in MODELS_TO_TRAIN:
    try:
        res = train_single_model(
            model_name=model_name,
            upscale_factor=UPSCALE_FACTOR,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE
        )
        benchmark_results.append(res)
    except Exception as e:
        print(f\"❌ [ERROR] Training failed for {model_name}: {e}\")
        import traceback
        traceback.print_exc()"""
    add_cell("code", c9)

    # -------------------------------------------------------------
    # Cell 10: Comparison Table & Analysis
    # -------------------------------------------------------------
    c10 = """# ==========================================================
# BENCHMARK COMPARISON TABLE (SCALE 2X)
# ==========================================================
summary_data = []

for r in benchmark_results:
    m_name = r[\"model_name\"]
    temp_m = build_model(m_name, upscale_factor=UPSCALE_FACTOR)
    param_count = sum(p.numel() for p in temp_m.parameters())
    summary_data.append({
        \"Model\": m_name,
        \"Parameters\": f\"{param_count:,}\",
        \"Best Epoch\": r[\"best_epoch\"],
        \"Best PSNR (dB)\": f\"{r['best_psnr']:.2f}\",
        \"Best SSIM\": f\"{r['best_ssim']:.4f}\",
        \"Training Time (min)\": f\"{r['training_time_min']:.2f}\",
        \"Weights Path\": os.path.basename(r[\"best_weights_path\"])
    })

df_summary = pd.DataFrame(summary_data)
print(\"\\n\" + \"=\" * 75)
print(f\"   NIH CHESTX-RAY 2X SUPER-RESOLUTION BENCHMARK SUMMARY (10 EPOCHS)\")
print(\"=\" * 75)
print(df_summary.to_string(index=False))

# Export to CSV
csv_path = os.path.join(WEIGHTS_DIR, \"benchmark_summary_2x.csv\")
df_summary.to_csv(csv_path, index=False)
print(f\"\\n[INFO] Summary exported to: {csv_path}\")"""
    add_cell("code", c10)

    # -------------------------------------------------------------
    # Cell 11: Visual Comparison Plot (HR vs Bicubic vs 6 Models)
    # -------------------------------------------------------------
    c11 = """# ==========================================================
# VISUAL COMPARISON (HR vs BICUBIC vs 6 SR MODELS)
# ==========================================================
test_lr, test_hr = val_dataset[0]
test_lr_tensor = test_lr.unsqueeze(0).to(DEVICE)
test_hr_tensor = test_hr.unsqueeze(0).to(DEVICE)

# Bicubic baseline
bicubic_sr = F.interpolate(test_lr_tensor, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)
bicubic_psnr = calculate_psnr(bicubic_sr, test_hr_tensor)
bicubic_ssim = calculate_ssim(bicubic_sr, test_hr_tensor)

predictions = {
    \"HR Ground Truth\": (test_hr.permute(1, 2, 0).cpu().numpy(), None, None),
    \"Bicubic Baseline\": (bicubic_sr.squeeze(0).permute(1, 2, 0).cpu().numpy(), bicubic_psnr, bicubic_ssim)
}

for r in benchmark_results:
    m_name = r[\"model_name\"]
    ckpt_path = r[\"best_weights_path\"]
    if os.path.exists(ckpt_path):
        m = build_model(m_name, upscale_factor=UPSCALE_FACTOR).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        m.load_state_dict(ckpt[\"state_dict\"])
        m.eval()
        with torch.no_grad():
            sr = m(test_lr_tensor)
            p = calculate_psnr(sr, test_hr_tensor)
            s = calculate_ssim(sr, test_hr_tensor)
            predictions[m_name] = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy(), p, s)

# Plot comparison grid
num_plots = len(predictions)
cols = 4
rows = math.ceil(num_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
axes = axes.flatten()

for idx, (title, (img_arr, p_val, s_val)) in enumerate(predictions.items()):
    ax = axes[idx]
    ax.imshow(np.clip(img_arr, 0.0, 1.0))
    if p_val is not None:
        ax.set_title(f\"{title}\\n{p_val:.2f} dB | SSIM: {s_val:.4f}\", fontsize=11, fontweight='bold')
    else:
        ax.set_title(f\"{title}\\nGround Truth\", fontsize=11, fontweight='bold')
    ax.axis(\"off\")

for j in range(num_plots, len(axes)):
    axes[j].axis(\"off\")

plt.tight_layout()
visual_comp_path = os.path.join(WEIGHTS_DIR, f\"visual_comparison_2x.png\")
plt.savefig(visual_comp_path, dpi=200)
plt.show()
print(f\"[INFO] Saved visual comparison to: {visual_comp_path}\")"""
    add_cell("code", c11)

    # -------------------------------------------------------------
    # Cell 12: Kaggle One-Click ZIP Export
    # -------------------------------------------------------------
    c12 = """# ==========================================================
# ARCHIVE & DOWNLOAD WEIGHTS (KAGGLE ONE-CLICK EXPORT)
# ==========================================================
zip_output_path = os.path.join(WORKING_DIR, f\"sr_models_weights_2x.zip\")

print(f\"[INFO] Packaging all weights from '{WEIGHTS_DIR}' into ZIP...\")
with zipfile.ZipFile(zip_output_path, \"w\", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(WEIGHTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, WEIGHTS_DIR)
            zipf.write(file_path, arcname)

zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
print(f\"[SUCCESS] Archive ready: {zip_output_path} ({zip_size_mb:.2f} MB)\")
print(\"👉 You can now download this ZIP file directly from the Kaggle Output tab in the right sidebar!\")"""
    add_cell("code", c12)

    # Write notebook file
    target_path = "train_models_2x.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[SUCCESS] Successfully written {len(nb['cells'])} cells to {target_path}")

if __name__ == "__main__":
    create_2x_notebook()
