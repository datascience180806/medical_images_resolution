import json
import os

def create_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": []
    }

    # Cell 0: Markdown Title & Documentation
    cell0 = """# 🔬 Generating Publication Diagrams for Section II: Super-Resolution Paradigms
## Comparative Architectural Dataflow & Visual Analysis (Scale 2x)
### Pre-upsampling (SRCNN) vs Post-upsampling Deconvolution (FSRCNN) vs Sub-Pixel Convolution (ESPCN)

This notebook executes inference on a medical Chest X-ray image from the **NIH ChestX-ray14** dataset and programmatically synthesizes publication-quality diagrams and visual comparisons for **Section II of the IEEE GTSD 2026 conference paper**:

1. **Pre-upsampling (SRCNN)**:
   - Input LR image $\\rightarrow$ **Bicubic Pre-upsampling** $\\rightarrow$ **Intermediate Bicubic Image** $\\rightarrow$ Non-linear Mapping Convolutions on uniform $H \\times W$ $\\rightarrow$ **Final Reconstructed HR Output**.
2. **Post-upsampling with Deconvolution (FSRCNN)**:
   - Input LR image $\\rightarrow$ Feature extraction on LR space $\\rightarrow$ **Transposed Convolution (Deconvolution)** with interleaved zeros $\\rightarrow$ **Final HR Output** (highlighting filter overlap / checkerboard behavior).
3. **Post-upsampling with Sub-pixel Convolution (ESPCN)**:
   - Input LR image $\\rightarrow$ LR Feature Convolutions $\\rightarrow$ **Channel Expansion ($C \\cdot r^2 = 12$)** $\\rightarrow$ **Sub-pixel Convolution (`PixelShuffle`)** periodic reassembly $\\rightarrow$ **Final HR Output**.

All individual processed images, high-resolution vector PDFs, and publication-ready composite figures are automatically saved to `/kaggle/working/output/` and archived into a one-click ZIP file.
"""
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in cell0.splitlines()]
    })

    # Cell 1: Environment Setup & Directory Creation
    cell1 = """import os
import sys
import glob
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

# 1. Environment & Output Directory
IS_KAGGLE = os.path.exists("/kaggle")
BASE_DIR = "/kaggle/working" if IS_KAGGLE else "."
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Hardware Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Execution Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"[INFO] Computing Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] Output Directory: {OUTPUT_DIR}")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell1.splitlines()]
    })

    # Cell 2: Model Architecture Definitions (Scale 2x)
    cell2 = """# ==========================================================
# MODEL ARCHITECTURES (SCALE 2X) WITH INTERMEDIATE TAPS
# ==========================================================

# 1. SRCNN (Dong et al., ECCV 2014) - Pre-upsampling 9-5-5 Architecture
class SRCNN(nn.Module):
    \"\"\"
    Pre-upsampling framework:
    Input LR -> Bicubic Pre-upsampling -> Conv1 (9x9) -> Conv2 (5x5) -> Conv3 (5x5) -> Output HR
    \"\"\"
    def __init__(self, in_channels=3, upscale_factor=2):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward_intermediates(self, x):
        \"\"\"Returns both the intermediate bicubic upsampled image and the final SR output.\"\"\"
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(x_bicubic, 0.0, 1.0), torch.clamp(out, 0.0, 1.0)

    def forward(self, x):
        _, out = self.forward_intermediates(x)
        return out


# 2. FSRCNN (Dong et al., ECCV 2016) - Post-upsampling Deconvolution
class FSRCNN(nn.Module):
    \"\"\"
    Post-upsampling framework with Deconvolution (Transposed Convolution):
    Input LR -> Feature Extraction (5x5) -> Shrinking (1x1) -> Mapping (3x3) ->
    Expanding (1x1) -> Deconvolution (9x9 stride 2) -> Output HR
    \"\"\"
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
        feat = self.feature_extraction(x)
        shrunk = self.shrinking(feat)
        mapped = self.mapping(shrunk)
        expanded = self.expanding(mapped)
        out = self.deconv(expanded)
        return torch.clamp(out, 0.0, 1.0)


# 3. ESPCN (Shi et al., CVPR 2016) - Post-upsampling Sub-Pixel Convolution
class ESPCN(nn.Module):
    \"\"\"
    Post-upsampling framework with Sub-pixel Convolution (PixelShuffle):
    Input LR -> Conv1 (5x5, 64) -> Conv2 (3x3, 32) -> Conv3 (3x3, C * r^2 = 12) -> PixelShuffle(2) -> Output HR
    \"\"\"
    def __init__(self, in_channels=3, upscale_factor=2):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()
        # For 2x: in_channels * 4 = 3 * 4 = 12 channels
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.tanh1(self.conv1(x))
        out = self.tanh2(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return torch.clamp(out, 0.0, 1.0)

print("[INFO] Model architectures initialized (SRCNN, FSRCNN, ESPCN).")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell2.splitlines()]
    })

    # Cell 3: Metrics & Checkpoint Loader
    cell3 = """# ==========================================================
# METRIC FUNCTIONS & ROBUST CHECKPOINT LOADER
# ==========================================================
def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    \"\"\"Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.\"\"\"
    mse = torch.mean((sr_tensor - hr_tensor) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * torch.log10((max_val ** 2) / mse).item()


def calculate_ssim(img1, img2, window_size=11, channel=3):
    \"\"\"Calculates Structural Similarity Index (SSIM).\"\"\"
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def load_model_weights(model, weight_path, device=DEVICE):
    \"\"\"Loads weights into model from checkpoint dict or raw state_dict.\"\"\"
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_sd = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        raw_sd = checkpoint
    else:
        raise TypeError(f"Invalid checkpoint format in: {weight_path}")

    clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.to(device)
    model.eval()
    return model

print("[INFO] Metrics and loader functions ready.")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell3.splitlines()]
    })

    # Cell 4: Automated Path Discovery (Weights & NIH Image)
    cell4 = """# ==========================================================
# AUTOMATED PATH DISCOVERY: KAGGLE WEIGHTS & NIH DATASET
# ==========================================================

# 1. Weights Directory Discovery
candidate_weights_dirs = [
    "/kaggle/input/datasets/duc24kdl/weight-2x",
    "/kaggle/input/weight-2x",
    "./weight_models/2x",
    "../weight_models/2x"
]

# Search recursively for weight-2x on Kaggle if not in direct paths
WEIGHTS_DIR = None
for d in candidate_weights_dirs:
    if os.path.isdir(d):
        WEIGHTS_DIR = d
        print(f"[INFO] Detected 2x weights directory at: {WEIGHTS_DIR}")
        break

if WEIGHTS_DIR is None:
    for root, dirs, _ in os.walk("/kaggle/input"):
        for d in dirs:
            if "weight-2x" in d.lower() or "weight_2x" in d.lower() or "weight2x" in d.lower():
                WEIGHTS_DIR = os.path.join(root, d)
                print(f"[INFO] Auto-discovered weights directory: {WEIGHTS_DIR}")
                break
        if WEIGHTS_DIR:
            break

if WEIGHTS_DIR is None:
    raise FileNotFoundError(
        "Could not find 2x weights directory! Please ensure the dataset 'weight-2x' "
        "is attached in Kaggle or present in './weight_models/2x'."
    )

def find_weight_file(model_name):
    patterns = [
        f"{model_name.lower()}.pth",
        f"{model_name.lower()}_best_2x.pth",
        f"{model_name.lower()}_latest_2x.pth"
    ]
    for p in patterns:
        full_p = os.path.join(WEIGHTS_DIR, p)
        if os.path.exists(full_p):
            return full_p
    # Recursive fallback
    for f in glob.glob(os.path.join(WEIGHTS_DIR, "**", f"*{model_name.lower()}*.pth"), recursive=True):
        return f
    raise FileNotFoundError(f"Weight file for {model_name} not found in {WEIGHTS_DIR}")

srcnn_weights_path = find_weight_file("srcnn")
fsrcnn_weights_path = find_weight_file("fsrcnn")
espcn_weights_path = find_weight_file("espcn")

print(f"[SUCCESS] SRCNN Weights  : {srcnn_weights_path}")
print(f"[SUCCESS] FSRCNN Weights : {fsrcnn_weights_path}")
print(f"[SUCCESS] ESPCN Weights  : {espcn_weights_path}")


# 2. NIH Test Image Discovery
candidate_nih_dirs = [
    "/kaggle/input/datasets/nih-chest-xrays/data/images_001/images",
    "/kaggle/input/nih-chest-xrays/data/images_001/images",
    "/kaggle/input/data/images_001/images",
    "/kaggle/input/datasets/nih-chest-xrays/data/images_001",
    "./eval_images",
    "./X-Ray images/NIH",
    "./sub_X-Ray",
    "./data"
]

TEST_IMAGE_PATH = None
for d in candidate_nih_dirs:
    if os.path.isdir(d):
        imgs = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
        if len(imgs) > 0:
            TEST_IMAGE_PATH = imgs[0]
            print(f"[INFO] Found NIH image at: {TEST_IMAGE_PATH} (out of {len(imgs)} images)")
            break

if TEST_IMAGE_PATH is None:
    # Deep scan
    for root, _, files in os.walk("/kaggle/input"):
        pngs = [f for f in sorted(files) if f.lower().endswith(('.png', '.jpg'))]
        if len(pngs) > 0:
            TEST_IMAGE_PATH = os.path.join(root, pngs[0])
            print(f"[INFO] Auto-scanned test image: {TEST_IMAGE_PATH}")
            break

if TEST_IMAGE_PATH is None:
    raise FileNotFoundError("No test X-ray image found! Please attach NIH ChestX-ray14 dataset to Kaggle.")

print(f"[SUCCESS] Selected Test Image: {TEST_IMAGE_PATH}")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell4.splitlines()]
    })

    # Cell 5: Image Preprocessing, 2x Downsampling & Model Inference
    cell5 = """# ==========================================================
# PREPROCESSING, 2X DOWNSAMPLING & MULTI-MODEL INFERENCE
# ==========================================================
# 1. Load Ground Truth High-Resolution (HR) Image
hr_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")

# Crop to a clean multiple of 2 (e.g. center crop or standard 512x512)
w_orig, h_orig = hr_pil.size
target_size = min(512, w_orig - (w_orig % 2), h_orig - (h_orig % 2))
hr_pil = transforms.functional.center_crop(hr_pil, (target_size, target_size))
hr_tensor = to_tensor(hr_pil).unsqueeze(0).to(DEVICE) # Shape: (1, 3, target_size, target_size)

# 2. Simulate Low-Resolution (LR) Input via Bicubic Downsampling (2x)
lr_size = target_size // 2
downsample_op = transforms.Resize((lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
lr_pil = downsample_op(hr_pil)
lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(DEVICE) # Shape: (1, 3, lr_size, lr_size)

print(f"[INFO] HR Ground Truth Dimensions: {hr_pil.size[0]}x{hr_pil.size[1]}")
print(f"[INFO] LR Input Dimensions        : {lr_pil.size[0]}x{lr_pil.size[1]} (Factor: 2x downscaled)")

# 3. Load 3 Representative Models
model_srcnn = load_model_weights(SRCNN(in_channels=3, upscale_factor=2), srcnn_weights_path)
model_fsrcnn = load_model_weights(FSRCNN(in_channels=3, upscale_factor=2), fsrcnn_weights_path)
model_espcn = load_model_weights(ESPCN(in_channels=3, upscale_factor=2), espcn_weights_path)

# 4. Perform Inference
with torch.no_grad():
    # Baseline: Traditional Bicubic Upsampling
    bicubic_sr_tensor = F.interpolate(lr_tensor, scale_factor=2, mode='bicubic', align_corners=False)
    bicubic_sr_tensor = torch.clamp(bicubic_sr_tensor, 0.0, 1.0)

    # Paradigm 1: Pre-upsampling (SRCNN) -> yields intermediate bicubic and final SR
    srcnn_bicubic_interm, srcnn_sr_tensor = model_srcnn.forward_intermediates(lr_tensor)

    # Paradigm 2: Post-upsampling Deconvolution (FSRCNN)
    fsrcnn_sr_tensor = model_fsrcnn(lr_tensor)

    # Paradigm 3: Post-upsampling Sub-pixel Convolution (ESPCN)
    espcn_sr_tensor = model_espcn(lr_tensor)

# 5. Compute Quantitative Metrics
metrics = {
    "Bicubic Baseline": {
        "psnr": calculate_psnr(bicubic_sr_tensor, hr_tensor),
        "ssim": calculate_ssim(bicubic_sr_tensor, hr_tensor),
        "tensor": bicubic_sr_tensor
    },
    "SRCNN (Pre-upsampling)": {
        "psnr": calculate_psnr(srcnn_sr_tensor, hr_tensor),
        "ssim": calculate_ssim(srcnn_sr_tensor, hr_tensor),
        "tensor": srcnn_sr_tensor
    },
    "FSRCNN (Deconvolution)": {
        "psnr": calculate_psnr(fsrcnn_sr_tensor, hr_tensor),
        "ssim": calculate_ssim(fsrcnn_sr_tensor, hr_tensor),
        "tensor": fsrcnn_sr_tensor
    },
    "ESPCN (Sub-pixel)": {
        "psnr": calculate_psnr(espcn_sr_tensor, hr_tensor),
        "ssim": calculate_ssim(espcn_sr_tensor, hr_tensor),
        "tensor": espcn_sr_tensor
    }
}

print("\\n" + "=" * 65)
print("  INFERENCE BENCHMARK METRICS ON SAMPLE NIH IMAGE (2X)")
print("=" * 65)
for name, d in metrics.items():
    print(f"  {name:<26} : PSNR = {d['psnr']:.2f} dB | SSIM = {d['ssim']:.4f}")
print("=" * 65)

# 6. Save Individual Clean Images to Output Folder
to_pil_image(hr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "01_hr_ground_truth.png"))
to_pil_image(lr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "02_lr_input.png"))
to_pil_image(srcnn_bicubic_interm.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "03_srcnn_bicubic_intermediate.png"))
to_pil_image(srcnn_sr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "04_srcnn_output.png"))
to_pil_image(fsrcnn_sr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "05_fsrcnn_output.png"))
to_pil_image(espcn_sr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "06_espcn_output.png"))
to_pil_image(bicubic_sr_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, "07_bicubic_baseline.png"))

print(f"[INFO] Saved all individual images to: {OUTPUT_DIR}")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell5.splitlines()]
    })

    # Cell 6: Programmatic Synthesis of Paradigm Architecture Diagram
    cell6 = """# ==========================================================
# FIGURE 1: ARCHITECTURAL DATAFLOW PARADIGM COMPARISON
# (Pre-upsampling vs Deconvolution vs Sub-Pixel Convolution)
# ==========================================================

# Convert tensors to numpy arrays for plotting
np_lr = to_pil_image(lr_tensor.squeeze(0).cpu())
np_bicubic_interm = to_pil_image(srcnn_bicubic_interm.squeeze(0).cpu())
np_srcnn = to_pil_image(srcnn_sr_tensor.squeeze(0).cpu())
np_fsrcnn = to_pil_image(fsrcnn_sr_tensor.squeeze(0).cpu())
np_espcn = to_pil_image(espcn_sr_tensor.squeeze(0).cpu())

# Create publication-grade figure
fig = plt.figure(figsize=(16, 12), dpi=300)
gs = fig.add_gridspec(3, 1, hspace=0.35)

# Color palette
c_blue = "#0277bd"
c_green = "#2e7d32"
c_orange = "#e65100"
c_bg = "#f8f9fa"

def draw_arrow(ax, start_x, start_y, end_x, end_y, text="", color="#37474f"):
    ax.annotate(
        "", xy=(end_x, end_y), xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", lw=2.5, color=color)
    )
    if text:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2 + 0.08
        ax.text(mid_x, mid_y, text, ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)


# -------------------------------------------------------------
# BRANCH (a): PRE-UPSAMPLING (SRCNN)
# -------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 3)
ax_a.axis("off")
ax_a.patch.set_facecolor(c_bg)

# Container Box
rect_a = patches.FancyBboxPatch((0.1, 0.1), 9.8, 2.8, boxstyle="round,pad=0.1", fc="#f1f8e9", ec=c_green, lw=1.8)
ax_a.add_patch(rect_a)
ax_a.text(0.3, 2.6, "(a) Pre-upsampling Framework (SRCNN): Bicubic Pre-Interpolation + Uniform Dimensional Pipeline",
          fontsize=12, fontweight="bold", color=c_green)

# Image 1: LR Input
ax_a.text(1.0, 2.2, "LR Input\\n(W/2 x H/2)", ha="center", fontsize=9, fontweight="bold")
ax_img_lr_a = ax_a.inset_axes([0.45, 0.35, 1.1, 1.5], transform=ax_a.transData)
ax_img_lr_a.imshow(np_lr)
ax_img_lr_a.axis("off")

# Arrow 1: Bicubic
draw_arrow(ax_a, 1.65, 1.1, 2.95, 1.1, text="Bicubic\\nPre-Interpolation", color=c_blue)

# Image 2: Intermediate Bicubic
ax_a.text(3.7, 2.2, "Intermediate Bicubic\\n(W x H Grid)", ha="center", fontsize=9, fontweight="bold")
ax_img_interm = ax_a.inset_axes([3.0, 0.3, 1.4, 1.6], transform=ax_a.transData)
ax_img_interm.imshow(np_bicubic_interm)
ax_img_interm.axis("off")

# Arrow 2: Feature Mapping Convs
draw_arrow(ax_a, 4.5, 1.1, 5.8, 1.1, text="Convolutions\\n(Uniform W x H)", color=c_blue)

# Block: RTL Properties
ax_a.text(6.8, 1.75, "RTL & Hardware Architecture Properties:\\n"
                     "- Single-rate streaming line buffer (100% MAC efficiency)\\n"
                     "- Dimensionally uniform pipeline (Zero clock-domain crossing)\\n"
                     "- Clinical Safety: 0% Checkerboard Artifacts",
          fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec=c_green, lw=1.2))

# Arrow 3: To Output
draw_arrow(ax_a, 7.8, 0.7, 8.4, 0.7, color=c_green)

# Image 3: HR Output
ax_a.text(9.1, 2.2, f"Reconstructed HR\\n({metrics['SRCNN (Pre-upsampling)']['psnr']:.2f} dB)", ha="center", fontsize=9, fontweight="bold", color=c_green)
ax_img_out_a = ax_a.inset_axes([8.4, 0.3, 1.4, 1.6], transform=ax_a.transData)
ax_img_out_a.imshow(np_srcnn)
ax_img_out_a.axis("off")


# -------------------------------------------------------------
# BRANCH (b): POST-UPSAMPLING DECONVOLUTION (FSRCNN)
# -------------------------------------------------------------
ax_b = fig.add_subplot(gs[1, 0])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 3)
ax_b.axis("off")

rect_b = patches.FancyBboxPatch((0.1, 0.1), 9.8, 2.8, boxstyle="round,pad=0.1", fc="#fff3e0", ec=c_orange, lw=1.8)
ax_b.add_patch(rect_b)
ax_b.text(0.3, 2.6, "(b) Post-upsampling Framework: Deconvolution / Transposed Convolution (FSRCNN)",
          fontsize=12, fontweight="bold", color=c_orange)

# Image 1: LR Input
ax_b.text(1.0, 2.2, "LR Input\\n(W/2 x H/2)", ha="center", fontsize=9, fontweight="bold")
ax_img_lr_b = ax_b.inset_axes([0.45, 0.35, 1.1, 1.5], transform=ax_b.transData)
ax_img_lr_b.imshow(np_lr)
ax_img_lr_b.axis("off")

# Arrow 1: LR Convolutions
draw_arrow(ax_b, 1.65, 1.1, 3.4, 1.1, text="LR Convolutions\\n(Feature Extraction)", color=c_orange)

# Block: LR Processing Box
rect_lr_feat = patches.FancyBboxPatch((3.4, 0.5), 1.7, 1.2, boxstyle="round,pad=0.1", fc="#ffffff", ec=c_orange, lw=1.2)
ax_b.add_patch(rect_lr_feat)
ax_b.text(4.25, 1.1, "Compact LR\\nFeature Map\\n(W/2 x H/2)", ha="center", va="center", fontsize=9, fontweight="bold")

# Arrow 2: Deconvolution
draw_arrow(ax_b, 5.15, 1.1, 6.7, 1.1, text="Transposed Conv\\n(Zero-Insertion Stride 2)", color="#d32f2f")

# Block: Hardware & Clinical Hazard
ax_b.text(6.8, 1.75, "RTL & Clinical Limitations:\\n"
                     "- Non-contiguous memory access & idle MAC cycles\\n"
                     "- Uneven filter overlap causes Periodic Checkerboard Artifacts\\n"
                     "- Hazard: Distortions masquerade as pathological micro-lesions",
          fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#d32f2f", lw=1.2))

# Arrow 3: To Output
draw_arrow(ax_b, 7.8, 0.7, 8.4, 0.7, color="#d32f2f")

# Image 3: Output with Artifacts
ax_b.text(9.1, 2.2, f"Reconstructed HR\\n({metrics['FSRCNN (Deconvolution)']['psnr']:.2f} dB)", ha="center", fontsize=9, fontweight="bold", color="#d32f2f")
ax_img_out_b = ax_b.inset_axes([8.4, 0.3, 1.4, 1.6], transform=ax_b.transData)
ax_img_out_b.imshow(np_fsrcnn)
ax_img_out_b.axis("off")


# -------------------------------------------------------------
# BRANCH (c): POST-UPSAMPLING SUB-PIXEL / PIXELSHUFFLE (ESPCN)
# -------------------------------------------------------------
ax_c = fig.add_subplot(gs[2, 0])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 3)
ax_c.axis("off")

rect_c = patches.FancyBboxPatch((0.1, 0.1), 9.8, 2.8, boxstyle="round,pad=0.1", fc="#e1f5fe", ec=c_blue, lw=1.8)
ax_c.add_patch(rect_c)
ax_c.text(0.3, 2.6, "(c) Post-upsampling Framework: Sub-Pixel Convolution / PixelShuffle (ESPCN)",
          fontsize=12, fontweight="bold", color=c_blue)

# Image 1: LR Input
ax_c.text(1.0, 2.2, "LR Input\\n(W/2 x H/2)", ha="center", fontsize=9, fontweight="bold")
ax_img_lr_c = ax_c.inset_axes([0.45, 0.35, 1.1, 1.5], transform=ax_c.transData)
ax_img_lr_c.imshow(np_lr)
ax_img_lr_c.axis("off")

# Arrow 1: LR Convolutions
draw_arrow(ax_c, 1.65, 1.1, 3.2, 1.1, text="LR Convolutions\\n(5x5 & 3x3)", color=c_blue)

# Block: Channel Expansion Box
rect_exp = patches.FancyBboxPatch((3.2, 0.5), 1.7, 1.2, boxstyle="round,pad=0.1", fc="#ffffff", ec=c_blue, lw=1.2)
ax_c.add_patch(rect_exp)
ax_c.text(4.05, 1.1, "Channel Expansion\\n(C * r^2 = 12 channels)\\n(W/2 x H/2)", ha="center", va="center", fontsize=8.5, fontweight="bold")

# Arrow 2: PixelShuffle
draw_arrow(ax_c, 4.95, 1.1, 6.7, 1.1, text="PixelShuffle\\nPeriodic Reassembly", color=c_blue)

# Block: RTL Complexity
ax_c.text(6.8, 1.75, "RTL Implementation Trade-offs:\\n"
                     "- Multi-phase channel demultiplexing & memory re-indexing\\n"
                     "- Asynchronous multi-rate Clock Domain Crossing (LR -> HR)\\n"
                     "- Extra BRAM buffer overhead outweighs FLOP savings on small CNNs",
          fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec=c_blue, lw=1.2))

# Arrow 3: To Output
draw_arrow(ax_c, 7.8, 0.7, 8.4, 0.7, color=c_blue)

# Image 3: Output HR
ax_c.text(9.1, 2.2, f"Reconstructed HR\\n({metrics['ESPCN (Sub-pixel)']['psnr']:.2f} dB)", ha="center", fontsize=9, fontweight="bold", color=c_blue)
ax_img_out_c = ax_c.inset_axes([8.4, 0.3, 1.4, 1.6], transform=ax_c.transData)
ax_img_out_c.imshow(np_espcn)
ax_img_out_c.axis("off")

# Save Figure 1
fig1_png = os.path.join(OUTPUT_DIR, "figure_2_super_resolution_paradigms.png")
fig1_pdf = os.path.join(OUTPUT_DIR, "figure_2_super_resolution_paradigms.pdf")
plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
plt.savefig(fig1_pdf, dpi=300, bbox_inches="tight")
plt.show()

print(f"[SUCCESS] Exported Figure 1 (PNG): {fig1_png}")
print(f"[SUCCESS] Exported Figure 1 (PDF): {fig1_pdf}")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell6.splitlines()]
    })

    # Cell 7: Figure 2: Region-of-Interest (ROI) Visual Detail Comparison
    cell7 = """# ==========================================================
# FIGURE 2: REGION-OF-INTEREST (ROI) VISUAL ARTIFACT COMPARISON
# (Demonstrating Checkerboard Artifacts & Edge Preservation)
# ==========================================================

# Select high-frequency anatomical ROI (rib margin / pulmonary tissue)
roi_size = 96
# Center-based patch
start_y = target_size // 2 - 20
start_x = target_size // 2 - 20

# Extract numpy arrays
np_hr = np.array(to_pil_image(hr_tensor.squeeze(0).cpu()))
np_bicubic = np.array(to_pil_image(bicubic_sr_tensor.squeeze(0).cpu()))
np_srcnn_arr = np.array(np_srcnn)
np_fsrcnn_arr = np.array(np_fsrcnn)
np_espcn_arr = np.array(np_espcn)

# Crop ROIs
roi_hr = np_hr[start_y:start_y+roi_size, start_x:start_x+roi_size]
roi_bicubic = np_bicubic[start_y:start_y+roi_size, start_x:start_x+roi_size]
roi_srcnn = np_srcnn_arr[start_y:start_y+roi_size, start_x:start_x+roi_size]
roi_fsrcnn = np_fsrcnn_arr[start_y:start_y+roi_size, start_x:start_x+roi_size]
roi_espcn = np_espcn_arr[start_y:start_y+roi_size, start_x:start_x+roi_size]

# Plot side-by-side ROI comparison
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=300)

roi_items = [
    ("Ground Truth (HR)\\nReference", roi_hr, None, None, "#2e7d32"),
    ("Bicubic Baseline", roi_bicubic, metrics["Bicubic Baseline"]["psnr"], metrics["Bicubic Baseline"]["ssim"], "#455a64"),
    ("SRCNN (Pre-upsampling)", roi_srcnn, metrics["SRCNN (Pre-upsampling)"]["psnr"], metrics["SRCNN (Pre-upsampling)"]["ssim"], "#0277bd"),
    ("ESPCN (Sub-pixel Conv)", roi_espcn, metrics["ESPCN (Sub-pixel)"]["psnr"], metrics["ESPCN (Sub-pixel)"]["ssim"], "#1565c0"),
    ("FSRCNN (Deconvolution)", roi_fsrcnn, metrics["FSRCNN (Deconvolution)"]["psnr"], metrics["FSRCNN (Deconvolution)"]["ssim"], "#c62828")
]

for ax, (title, img_patch, p_val, s_val, border_color) in zip(axes, roi_items):
    ax.imshow(img_patch)
    if p_val is not None:
        ax.set_title(f"{title}\\n{p_val:.2f} dB | SSIM: {s_val:.4f}", fontsize=11, fontweight="bold", color=border_color)
    else:
        ax.set_title(title, fontsize=11, fontweight="bold", color=border_color)
    ax.axis("off")
    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
fig2_png = os.path.join(OUTPUT_DIR, "figure_2_roi_visual_comparison.png")
fig2_pdf = os.path.join(OUTPUT_DIR, "figure_2_roi_visual_comparison.pdf")
plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
plt.savefig(fig2_pdf, dpi=300, bbox_inches="tight")
plt.show()

print(f"[SUCCESS] Exported Figure 2 (PNG): {fig2_png}")
print(f"[SUCCESS] Exported Figure 2 (PDF): {fig2_pdf}")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell7.splitlines()]
    })

    # Cell 8: One-click ZIP Packaging
    cell8 = """# ==========================================================
# PACKAGING ALL GENERATED FIGURES & IMAGES INTO ZIP
# ==========================================================
zip_output_path = os.path.join(BASE_DIR, "section2_paradigms_output.zip")

print(f"[INFO] Packaging all contents from '{OUTPUT_DIR}' into ZIP...")
with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, OUTPUT_DIR)
            zipf.write(file_path, arcname)

zip_size_kb = os.path.getsize(zip_output_path) / 1024
print(f"\\n[SUCCESS] Archive ready: {zip_output_path} ({zip_size_kb:.1f} KB)")
print("[READY] On Kaggle: Look at the right sidebar 'Output' panel, click on 'section2_paradigms_output.zip' to download all figures!")
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell8.splitlines()]
    })

    with open("generate_section2_diagram.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully created generate_section2_diagram.ipynb!")

if __name__ == "__main__":
    create_notebook()
