# Kế hoạch: Quantize Swift-SRGAN Generator sang INT8 cho FPGA Zynq

## 1. Mục tiêu dự án

Quantize mạng **Generator** của Swift-SRGAN từ `float32` sang `int8` để trích xuất trọng số triển khai lên sơ đồ phần cứng FPGA Zynq (PS + PL) theo kiến trúc:

```
PC (Python) → Zynq PS (ARM - AXI DMA) → Zynq PL (FPGA Accelerator) → Zynq PS → PC (Python)
```

---

## 2. Hiện trạng & Kết quả đã đạt được

### 2.1. Đánh giá Baseline Model (FP32) trên Kaggle
Chạy trên 110 ảnh `eval_images/`:

| Chỉ số | Kết quả FP32 Baseline |
|--------|-----------------------|
| **Mean PSNR** | **41.8503 dB** |
| **Mean SSIM** | **0.9697** |

---

## 3. Các bước thực thi Lượng tử hóa Tối ưu (Per-Channel & Selective Quantization)

### Bước 1: Pull mã nguồn mới nhất từ GitHub
Trong Kaggle Notebook:
```bash
%cd medical_images_resolution
!git pull origin main
```

---

### Bước 2: Chạy Quantization Tối ưu sang INT8
Script `scripts/quantize_model.py` đã được bổ sung 2 kỹ thuật tối ưu cốt lõi:
1. **Per-Channel Weight Quantization (`--per_channel`):** Tính toán Scale riêng biệt cho từng channel của Depthwise Conv, khắc phục hoàn toàn hiện tượng triệt tiêu trọng số channel nhỏ về 0.
2. **Selective Quantization (`--selective`):** Giữ nguyên độ chính xác cao FP32 cho lớp vào (`initial`) và lớp ra (`upsampler`, `final_conv` + `tanh`), chỉ Quantize **16 khối Residual Blocks** và khối **ConvBlock** trung gian (đúng mục tiêu nạp vào mạch tăng tốc FPGA Zynq PL).

**Lệnh thực thi trong Kaggle Notebook:**
```bash
!python scripts/quantize_model.py \
    --weights ./models/netG_4x_epoch5.pth.tar \
    --calib_dir ./calibration_images \
    --output ./models/netG_4x_quantized_int8_optimized.pth \
    --selective \
    --per_channel
```

---

### Bước 3: Đánh giá mô hình INT8 Tối ưu
Chạy lại script `scripts/evaluate_model.py` để kiểm tra độ chính xác sau khi nâng cấp thuật toán Quantize:

**Lệnh thực thi trong Kaggle Notebook:**
```bash
!python scripts/evaluate_model.py \
    --eval_dir ./eval_images \
    --weights ./models/netG_4x_quantized_int8_optimized.pth \
    --output_csv ./logs/quantized_int8_optimized_metrics.csv
```
