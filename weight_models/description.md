# 📋 Hướng Dẫn Kỹ Thuật & Mô Tả Trọng Số Các Mô Hình Super-Resolution (2x)

Tài liệu này tổng hợp thông số kỹ thuật, cấu trúc mạng, định dạng checkpoint và mã nguồn suy luận (inference) cho các mô hình **Single Image Super-Resolution (SISR)** đã được huấn luyện trên tập dữ liệu ảnh y tế **NIH ChestX-ray14** ở tỉ lệ phóng đại **x2** và lưu tại thư mục [`weight_models/2x/`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x).

> [!NOTE]
> **Định dạng checkpoint:**
> Toàn bộ các file `.pth` trong thư mục `weight_models/2x/` đều được lưu dưới dạng **Checkpoint Dict** chứa đầy đủ metadata (`model_name`, `upscale_factor`, `epoch`, `best_epoch`, `best_psnr`, `best_ssim`, `state_dict`, `history`). Trọng số mạng `state_dict` đã được làm sạch (không chứa tiền tố `module.`), sẵn sàng load trực tiếp trên cả CPU và GPU.

---

## 1. Bảng Tổng Hợp Trọng Số & Trạng Thái Mô Hình (Scale 2x)

| Tên File Trọng Số | Tên Mô Hình | Dung Lượng | Số Tham Số | Định Dạng File | Epoch Đạt Được | Best PSNR (dB) | Best SSIM | Cơ Chế Upsampling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`srcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/srcnn.pth) | **SRCNN** | 0.28 MB | 69,251 | Checkpoint Dict | **Epoch 10/10** | **43.01 dB** | **0.9736** | Pre-upsampling (Bicubic $\times 2$ nội suy trước) |
| [`espcn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/espcn.pth) | **ESPCN** | 0.11 MB | 26,796 | Checkpoint Dict | **Epoch 10/10** | **37.97 dB** | **0.9633** | Post-upsampling (PixelShuffle / Sub-pixel Conv) |
| [`fsrcnn.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/fsrcnn.pth) | **FSRCNN** | 0.10 MB | 24,683 | Checkpoint Dict | **Epoch 10/10** | **27.62 dB** | **0.9169** | Post-upsampling (ConvTranspose2d / Deconvolution) |
| [`vdsr.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/vdsr.pth) | **VDSR** | 2.55 MB | 668,227 | Checkpoint Dict | **Epoch 1 (Best)** | **45.34 dB** | **0.9772** | Pre-upsampling + Global Residual Learning (20 layers) |
| [`srgan.pth`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/weight_models/2x/srgan.pth) | **SRGAN** | 5.43 MB | ~1.40M | Checkpoint Dict | **Epoch 10/10** | **42.00 dB** | **0.9655** | Post-upsampling (PixelShuffle) + 16 Residual Blocks |

*Ghi chú:*
* **VDSR:** File lưu checkpoint tốt nhất đạt PSNR cực cao ($45.34\text{ dB}$, SSIM $0.9772$).
* **EDSR:** Đã hoàn thành 10 epochs trên Kaggle đạt PSNR $42.60\text{ dB}$, SSIM $0.9787$. Khi bạn bổ sung `edsr.pth`, mô hình sử dụng cấu hình 8 Residual Blocks.

---

## 2. Định Nghĩa Kiến Trúc Mạng (PyTorch Model Classes)

Để nạp trọng số thành công, mô hình trong mã nguồn inference phải khớp 100% với định nghĩa lớp sau:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. SRCNN (Dong et al., ECCV 2014)
# ====================================================================
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
        # Nội suy Bicubic lên kích thước HR trước khi trích xuất đặc trưng
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 2. ESPCN (Shi et al., CVPR 2016)
# ====================================================================
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


# ====================================================================
# 3. FSRCNN (Dong et al., ECCV 2016)
# ====================================================================
class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        # 1. Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        # 2. Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        # 3. Non-linear mapping
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)
        # 4. Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        # 5. Deconvolution
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
# 4. VDSR (Kim et al., CVPR 2016)
# ====================================================================
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


# ====================================================================
# 5. EDSR (Lim et al., CVPRW 2017)
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
    def __init__(self, in_channels=3, upscale_factor=2, n_feats=64, n_resblocks=8, res_scale=0.1):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[EDSRResBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

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
```

---

## 3. Hàm Chuẩn Hóa Load Trọng Số (Unified Checkpoint Loader)

Vì các file checkpoint có thể chứa cả từ khóa metadata (`epoch`, `state_dict`, ...) hoặc là `state_dict` thuần túy, và có khả năng chứa tiền tố `module.` (nếu train bằng `nn.DataParallel`), người chạy inference **bắt buộc** sử dụng hàm nạp trọng số an toàn dưới đây:

```python
import os
import torch
import torch.nn as nn

def load_sr_model(model_name: str, weight_path: str, device: str = "cpu") -> nn.Module:
    """
    Tự động nhận diện kiến trúc, làm sạch tiền tố 'module.' và nạp trọng số an toàn.
    """
    model_name = model_name.upper()
    upscale_factor = 2

    # 1. Khởi tạo instance mô hình
    if model_name == "SRCNN":
        model = SRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "ESPCN":
        model = ESPCN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "FSRCNN":
        model = FSRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "VDSR":
        model = VDSR(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "EDSR":
        model = EDSR(in_channels=3, upscale_factor=upscale_factor, n_feats=64, n_resblocks=8)
    elif model_name == "SRGAN":
        model = SRGAN(in_channels=3, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    # 2. Đọc file checkpoint
    checkpoint = torch.load(weight_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        best_psnr = checkpoint.get("best_psnr", "N/A")
        print(f"[INFO] Loaded {model_name} from Checkpoint Dict (Epoch: {epoch}, Best PSNR: {best_psnr})")
    elif isinstance(checkpoint, dict):
        raw_state_dict = checkpoint
        print(f"[INFO] Loaded {model_name} from pure state_dict")
    else:
        raise TypeError("Định dạng file .pth không hợp lệ!")

    # 3. Loại bỏ tiền tố 'module.' nếu có
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        new_key = k.replace("module.", "")
        clean_state_dict[new_key] = v

    # 4. Gán trọng số vào mô hình
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
```

---

## 4. Quy Chuẩn Dữ Liệu Vào / Ra (Input / Output Specifications)

* **Tensor đầu vào (`lr_tensor`):**
  * **Kích thước:** `(B, 3, H, W)` — Thường $B=1$ khi chạy đơn lẻ từng ảnh.
  * **Không gian màu:** RGB (Nếu ảnh X-Ray là ảnh xám 1 kênh Grayscale, cần dùng `img.convert("RGB")` hoặc duplicate thành 3 kênh).
  * **Miền giá trị:** Float32 chuẩn hóa trong khoảng $[0.0, 1.0]$.
  * *Lưu ý quan trọng:* Tất cả 5 model đều nhận trực tiếp ảnh **Low-Resolution (LR)** đầu vào. Mô hình SRCNN và VDSR đã được thiết kế sẵn hàm `F.interpolate(mode='bicubic')` bên trong `forward()`, người dùng **không cần** tự phóng to ảnh thủ công trước khi đưa vào mô hình.
* **Tensor đầu ra (`sr_tensor`):**
  * **Kích thước:** `(B, 3, 2H, 2W)` tương ứng với tỉ lệ phóng đại $\times 2$.
  * **Miền giá trị:** Tự động kẹp (clamped) trong đoạn $[0.0, 1.0]$.

---

## 5. Script Mẫu Chạy Toàn Bộ Quá Trình Inference (Full Example Script)

Bạn của bạn có thể lưu đoạn code sau thành file `run_inference.py` để test ngay lập tức:

```python
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

# Giả sử file chứa các class kiến trúc ở Mục 2 tên là models.py
# from models import load_sr_model

def run_single_image_inference(image_path: str, model_name: str, weight_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Khởi tạo mô hình {model_name} trên thiết bị: {device}")
    
    model = load_sr_model(model_name, weight_path, device=device)

    # Đọc ảnh Low-Resolution
    lr_pil = Image.open(image_path).convert("RGB")
    lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device) # Shape: (1, 3, H, W)

    with torch.no_grad():
        sr_tensor = model(lr_tensor) # Shape: (1, 3, 2*H, 2*W)

    # Chuyển về định dạng PIL Image và lưu
    sr_pil = to_pil_image(sr_tensor.squeeze(0).cpu())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sr_pil.save(output_path)
    print(f">> Đã lưu ảnh siêu phân giải tại: {output_path} (Kích thước: {sr_pil.size})")

if __name__ == "__main__":
    # Ví dụ minh họa chạy mô hình SRCNN
    model_choice = "SRCNN"
    weight_file = "weight_models/2x/srcnn.pth"
    test_image = "eval_images/00001255_011.png" # Hoặc bất kỳ ảnh nào trong eval_images
    result_image = "./results/srcnn_sr_2x.png"

    if os.path.exists(weight_file) and os.path.exists(test_image):
        run_single_image_inference(test_image, model_choice, weight_file, result_image)
    else:
        print("Vui lòng kiểm tra lại đường dẫn file trọng số hoặc ảnh đầu vào!")
```

---

## 6. Hướng Dẫn Tính Toán Chỉ Số Đo Đạc (PSNR & SSIM)

Khi đánh giá so sánh với ảnh High-Resolution (HR) gốc (Ground Truth), sử dụng hàm đo chuẩn dưới đây:

```python
import math
import torch
import torch.nn.functional as F

def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    """Tính PSNR (dB) giữa tensor SR và HR (cùng shape: B, 3, H, W)."""
    mse = torch.mean((sr_tensor - hr_tensor) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse.item())

def calculate_ssim(img1, img2, window_size=11):
    """Tính SSIM giữa hai tensor ảnh chuẩn hóa [0, 1]."""
    channel = img1.size(1)
    
    # Tạo Gaussian kernel
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
    return ssim_map.mean().item()
```
