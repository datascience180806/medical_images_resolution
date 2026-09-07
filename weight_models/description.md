# 📋 Hướng Dẫn Kỹ Thuật & Mô Tả Trọng Số Các Mô Hình Super-Resolution (Scale 2x, 3x & 4x)

Tài liệu này tổng hợp thông số kỹ thuật, cấu trúc mạng, định dạng checkpoint, số liệu benchmark và mã nguồn suy luận (inference) cho các mô hình **Single Image Super-Resolution (SISR)** đã được huấn luyện trên tập dữ liệu ảnh y tế **NIH ChestX-ray14** ở ba tỉ lệ phóng đại **2x**, **3x** và **4x**, lưu tại các thư mục:
* [`weight_models/2x/`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x)
* [`weight_models/3x/`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x)
* [`weight_models/4x/`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x)

> [!NOTE]
> **Định dạng checkpoint:**
> Toàn bộ các file `.pth` đều được lưu dưới dạng **Checkpoint Dict** chứa đầy đủ metadata (`model_name`, `upscale_factor`, `epoch`, `best_epoch`, `best_psnr`, `best_ssim`, `state_dict`). Trọng số mạng `state_dict` đã được làm sạch (không chứa tiền tố `module.` của `nn.DataParallel`), sẵn sàng nạp trực tiếp trên cả CPU và GPU.

---

## 1. Bảng Tổng Hợp Trọng Số & Trạng Thái Mô Hình (Scale 4x)

Dữ liệu được trích xuất trực tiếp từ kết quả benchmark 10 Epochs trên hệ thống 2x NVIDIA Tesla T4 GPUs (Kaggle) trên tập dữ liệu 12,000 ảnh NIH Chest X-Ray:

| Tên File Trọng Số | Tên Mô Hình | Dung Lượng | Số Tham Số | Định Dạng File | Best Epoch | Best PSNR (dB) | Best SSIM | Cơ Chế Phóng Đại (Upsampling) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`srcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/srcnn.pth) | **SRCNN** | 0.08 MB | 20,099 | Checkpoint Dict | **Epoch 10/10** | **41.3255 dB** | **0.9594** | Pre-upsampling (Bicubic $\times 4$ nội suy trước, $9\text{-}1\text{-}5$) |
| [`espcn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/espcn.pth) | **ESPCN** | 0.15 MB | 37,200 | Checkpoint Dict | **Epoch 10/10** | **20.2018 dB** | **0.6652** | Post-upsampling ($3 \times 4^2 = 48$ chs $\rightarrow$ `PixelShuffle(4)`) |
| [`fsrcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/fsrcnn.pth) | **FSRCNN** | 0.10 MB | 24,683 | Checkpoint Dict | **Epoch 10/10** | **14.1040 dB** | **0.6113** | Post-upsampling (`ConvTranspose2d` stride=4, out_pad=3) |
| [`vdsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/vdsr.pth) | **VDSR** | 2.56 MB | 668,227 | Checkpoint Dict | **Epoch 9/10** | **43.6998 dB** | **0.9607** | Pre-upsampling + Global Residual (20 layers Conv) |
| [`edsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/edsr.pth) | **EDSR** | 3.55 MB | 926,723 | Checkpoint Dict | **Epoch 10/10** | **39.7223 dB** | **0.9601** | Enhanced Residual (8 ResBlocks + 2x Cascaded `PixelShuffle(2)`) |
| [`srgan.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/4x/srgan.pth) | **SRGAN** | 6.00 MB | 1,548,547 | Checkpoint Dict | **Epoch 10/10** | **41.3291 dB** | **0.9502** | 16 Residual Blocks + 2x Cascaded `PixelShuffle(2)` Generator |

---

## 2. Bảng Tổng Hợp Trọng Số & Trạng Thái Mô Hình (Scale 3x)

| Tên File Trọng Số | Tên Mô Hình | Dung Lượng | Số Tham Số | Định Dạng File | Best Epoch | Best PSNR (dB) | Best SSIM | Cơ Chế Phóng Đại (Upsampling) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`srcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x/srcnn.pth) | **SRCNN** | 0.27 MB | 69,251 | Checkpoint Dict | **Epoch 10/10** | **43.3969 dB** | **0.9657** | Pre-upsampling (Bicubic $\times 3$ nội suy trước, $9\text{-}5\text{-}5$) |
| *`espcn.pth` (cần bổ sung)* | **ESPCN** | 0.12 MB | 31,131 | Checkpoint Dict | **Epoch 10/10** | **37.7182 dB** | **0.9526** | Post-upsampling ($3 \times 3^2 = 27$ chs $\rightarrow$ `PixelShuffle(3)`) |
| [`fsrcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x/fsrcnn.pth) | **FSRCNN** | 0.10 MB | 24,683 | Checkpoint Dict | **Epoch 10/10** | **17.8355 dB** | **0.6446** | Post-upsampling (`ConvTranspose2d` stride=3, out_pad=2) |
| [`vdsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x/vdsr.pth) | **VDSR** | 2.56 MB | 668,227 | Checkpoint Dict | **Epoch 10/10** | **44.5962 dB** | **0.9661** | Pre-upsampling + Global Residual (20 layers Conv) |
| [`edsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x/edsr.pth) | **EDSR** | 3.69 MB | 963,651 | Checkpoint Dict | **Epoch 10/10** | **41.6324 dB** | **0.9665** | Enhanced Residual (8 ResBlocks + `PixelShuffle(3)`) |
| [`srgan.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/3x/srgan.pth) | **SRGAN** | 6.14 MB | 1,585,411 | Checkpoint Dict | **Epoch 10/10** | **41.2789 dB** | **0.9532** | 16 Residual Blocks + `PixelShuffle(3)` Generator |

---

## 3. Bảng Tổng Hợp Trọng Số & Trạng Thái Mô Hình (Scale 2x)

| Tên File Trọng Số | Tên Mô Hình | Dung Lượng | Số Tham Số | Định Dạng File | Best Epoch | Best PSNR (dB) | Best SSIM | Cơ Chế Phóng Đại (Upsampling) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`srcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/srcnn.pth) | **SRCNN** | 0.28 MB | 69,251 | Checkpoint Dict | **Epoch 10/10** | **43.01 dB** | **0.9736** | Pre-upsampling (Bicubic $\times 2$ nội suy trước, $9\text{-}5\text{-}5$) |
| [`espcn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/espcn.pth) | **ESPCN** | 0.11 MB | 26,796 | Checkpoint Dict | **Epoch 10/10** | **37.97 dB** | **0.9633** | Post-upsampling ($3 \times 2^2 = 12$ chs $\rightarrow$ `PixelShuffle(2)`) |
| [`fsrcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/fsrcnn.pth) | **FSRCNN** | 0.10 MB | 24,683 | Checkpoint Dict | **Epoch 10/10** | **27.62 dB** | **0.9169** | Post-upsampling (`ConvTranspose2d` stride=2, out_pad=1) |
| [`vdsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/vdsr.pth) | **VDSR** | 2.55 MB | 668,227 | Checkpoint Dict | **Epoch 1/10** | **45.34 dB** | **0.9772** | Pre-upsampling + Global Residual (20 layers Conv) |
| *`edsr.pth` (cần bổ sung)* | **EDSR** | ~3.6 MB | 963,651 | Checkpoint Dict | **Epoch 10/10** | **42.60 dB** | **0.9787** | Enhanced Residual (8 ResBlocks + `PixelShuffle(2)`) |
| [`srgan.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/srgan.pth) | **SRGAN** | 5.43 MB | ~1.40M | Checkpoint Dict | **Epoch 10/10** | **42.00 dB** | **0.9655** | 16 Residual Blocks + `PixelShuffle(2)` Generator |

---

## 4. Bảng So Sánh Toàn Diện: Scale 2x vs Scale 3x vs Scale 4x

| Mô Hình | PSNR 2x (dB) | PSNR 3x (dB) | PSNR 4x (dB) | SSIM 2x | SSIM 3x | SSIM 4x | Nhận Định & Xu Hướng Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **VDSR** | **45.34** | **44.60** | **43.70** | **0.9772** | **0.9661** | **0.9607** | Đạt chất lượng giải phẫu cao nhất mọi thang đo nhờ 20 layers residual |
| **SRCNN** | 43.01 | 43.40 | 41.33 | 0.9736 | 0.9657 | 0.9594 | Pre-upsampling ổn định vượt trội, cực kỳ phù hợp triển khai FPGA Edge |
| **SRGAN** | 42.00 | 41.28 | 41.33 | 0.9655 | 0.9532 | 0.9502 | Tái tạo chi tiết sắc nét, PSNR ở 4x ngang ngửa SRCNN |
| **EDSR** | 42.60 | 41.63 | 39.72 | 0.9787 | 0.9665 | 0.9601 | Bảo toàn SSIM cao nhất ở 2x và 3x, cấu trúc sâu khử nhiễu tốt |
| **ESPCN** | 37.97 | 37.72 | 20.20 | 0.9633 | 0.9526 | 0.6652 | Ở 4x, PixelShuffle đơn tầng bị suy thoái nặng do dung lượng mạng quá nhỏ |
| **FSRCNN** | 27.62 | 17.84 | 14.10 | 0.9169 | 0.6446 | 0.6113 | Deconvolution bị lỗi chồng lấp bộ lọc (checkerboard) nghiêm trọng khi scale lớn |

---

## 5. Định Nghĩa Kiến Trúc Mạng Đa Tỉ Lệ (Multi-Scale PyTorch Models)

Để nạp thành công cả trọng số 2x, 3x và 4x, kiến trúc mạng hỗ trợ linh hoạt tham số `upscale_factor` như sau:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. SRCNN (Dong et al., ECCV 2014) - Pre-upsampling
# ====================================================================
class SRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        # Hỗ trợ cả 9-1-5 (20k params ở 4x) và 9-5-5 (69k params ở 2x/3x)
        conv2_kernel = 1 if upscale_factor == 4 else 5
        conv2_pad = 0 if upscale_factor == 4 else 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=conv2_kernel, padding=conv2_pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 2. ESPCN (Shi et al., CVPR 2016) - Sub-Pixel Convolution
# ====================================================================
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
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


# ====================================================================
# 3. FSRCNN (Dong et al., ECCV 2016) - Post-upsampling Deconvolution
# ====================================================================
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


# ====================================================================
# 4. VDSR (Kim et al., CVPR 2016) - 20-Layer Deep Residual Network
# ====================================================================
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


# ====================================================================
# 5. EDSR (Lim et al., CVPRW 2017) - Enhanced Deep Residual Network
# ====================================================================
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


# ====================================================================
# 6. SRGAN Generator (Ledig et al., CVPR 2017)
# ====================================================================
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
        if upscale_factor in [2, 4]:
            num_stages = int(math.log2(upscale_factor))
            stages = []
            for _ in range(num_stages):
                stages.extend([
                    nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU(num_parameters=num_channels)
                ])
            self.upsampler = nn.Sequential(*stages)
        else: # 3x
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
```

---

## 6. Hàm Nạp Trọng Số Thông Minh (Smart Unified Loader)

```python
import os
import torch
import torch.nn as nn

def load_sr_model(model_name: str, weight_path: str, upscale_factor: int = 4, device: str = "cpu") -> nn.Module:
    """
    Tự động nhận diện cấu hình, làm sạch tiền tố 'module.' và nạp trọng số an toàn.
    """
    model_name = model_name.upper()

    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "upscale_factor" in checkpoint:
        upscale_factor = checkpoint["upscale_factor"]

    if model_name == "SRCNN":
        model = SRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "ESPCN":
        model = ESPCN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "FSRCNN":
        model = FSRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "VDSR":
        model = VDSR(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "EDSR":
        model = EDSR(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "SRGAN":
        model = SRGAN(in_channels=3, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        best_psnr = checkpoint.get("best_psnr", "N/A")
        best_ssim = checkpoint.get("best_ssim", "N/A")
        print(f"[INFO] Loaded {model_name} (Scale {upscale_factor}x) from Checkpoint Dict (Epoch: {epoch}, PSNR: {best_psnr:.4f} dB, SSIM: {best_ssim:.4f})")
    elif isinstance(checkpoint, dict):
        raw_state_dict = checkpoint
        print(f"[INFO] Loaded {model_name} (Scale {upscale_factor}x) from pure state_dict")
    else:
        raise TypeError("Định dạng file .pth không hợp lệ!")

    clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}

    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
```

---

## 7. Ví Dụ Chạy Inference Cho Scale 4x

```python
import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

MODEL_NAME = "SRCNN"                     # "SRCNN" | "VDSR" | "EDSR" | "SRGAN" | "ESPCN" | "FSRCNN"
SCALE = 4
WEIGHT_PATH = f"weight_models/{SCALE}x/{MODEL_NAME.lower()}.pth"
TEST_IMAGE = "eval_images/00001255_011.png"
OUTPUT_IMAGE = f"./results/{MODEL_NAME.lower()}_sr_{SCALE}x.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_sr_model(MODEL_NAME, WEIGHT_PATH, upscale_factor=SCALE, device=device)

lr_img = Image.open(TEST_IMAGE).convert("RGB")
lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device) # Shape: (1, 3, H, W)

with torch.no_grad():
    sr_tensor = model(lr_tensor)                      # Shape: (1, 3, 4*H, 4*W)

sr_pil = to_pil_image(sr_tensor.squeeze(0).cpu())
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
sr_pil.save(OUTPUT_IMAGE)
print(f">> Thành công! Đã lưu ảnh siêu phân giải 4x tại: {OUTPUT_IMAGE} (Kích thước: {sr_pil.size})")
```
