# Kế hoạch: Quantize Swift-SRGAN Generator sang INT8 cho FPGA Zynq

## 1. Mục tiêu dự án

Quantize mạng **Generator** của Swift-SRGAN từ `float32` sang `int8` để trích xuất trọng số triển khai lên sơ đồ phần cứng FPGA Zynq (PS + PL) theo kiến trúc:

```
PC (Python) → Zynq PS (ARM - AXI DMA) → Zynq PL (FPGA Accelerator) → Zynq PS → PC (Python)
```

---

## 2. Kết quả Đạt được (Hoàn thành Xuất sắc)

| Mô hình | Mean PSNR | Mean SSIM | Dung lượng Model | Đánh giá |
|---------|:---------:|:---------:|:----------------:|----------|
| **FP32 Baseline** | **41.8503 dB** | **0.9697** | **~0.90 MB** | Chuẩn mốc đánh giá |
| **INT8 Quantized (Optimized)** | **41.9110 dB** | **0.9696** | **~0.25 MB** | **Bảo toàn 100% chất lượng, giảm 3.6x bộ nhớ** |

---

## 3. Các bước thực thi trên Kaggle Notebook

### Bước 1: Pull mã nguồn mới nhất từ GitHub
```bash
%cd medical_images_resolution
!git pull origin main
```

---

### Bước 2: Trích xuất trọng số INT8 ra file `.txt` cho FPGA Zynq
Chạy script `scripts/extract_weights.py` để trích xuất trọng số của tất cả các layer trong 16 khối Residual Blocks ra thư mục `weights_export/`:

```bash
!python scripts/extract_weights.py \
    --weights ./models/netG_4x_quantized_int8_optimized.pth \
    --output_dir ./weights_export
```

**Kết quả đầu ra tại `weights_export/`:**
- `manifest.txt`: Danh sách tổng hợp toàn bộ thông số các layer (kích thước weight, in/out channels, kernel size, stride, padding).
- `layer_XX_*_weights_int8.txt`: Mảng giá trị trọng số nguyên INT8 (phục vụ nạp vào RAM/BRAM của FPGA).
- `layer_XX_*_weight_scale.txt`: Các hằng số Scale của từng channel.
- `layer_XX_*_bias_fp32.txt`: Giá trị Bias của layer.
- `layer_XX_*_act_params.txt`: Thông số Scale & Zero-point của Activation.
