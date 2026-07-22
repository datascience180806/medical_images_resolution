# Kế hoạch: Quantize Swift-SRGAN Generator sang INT8 cho FPGA Zynq

## 1. Mục tiêu dự án

Quantize mạng **Generator** của Swift-SRGAN từ `float32` sang `int8` để trích xuất trọng số triển khai lên sơ đồ phần cứng FPGA Zynq (PS + PL) theo kiến trúc:

```
PC (Python) → Zynq PS (ARM - AXI DMA) → Zynq PL (FPGA Accelerator) → Zynq PS → PC (Python)
```

---

## 2. Hiện trạng & Kết quả đã đạt được

### 2.1. Đánh giá Baseline Model (FP32) trên Kaggle (ĐÃ HOÀN THÀNH)
Chạy trên 110 ảnh `eval_images/`:

| Chỉ số | Kết quả FP32 |
|--------|--------------|
| **Mean PSNR** | **41.8503 dB** |
| **Mean SSIM** | **0.9697** |

---

## 3. Các bước thực thi tiếp theo trên Kaggle

### Bước 1: Pull mã nguồn mới nhất từ GitHub
Trong Kaggle Notebook:
```bash
%cd medical_images_resolution
!git pull origin main
```

---

### Bước 2: Chạy Quantization sang INT8
Chạy script `scripts/quantize_model.py` để:
1. Fuse toàn bộ các lớp `Conv2d + BatchNorm2d` (chuẩn hóa kiến trúc cho hardware FPGA).
2. Chuyển đổi trọng số sang `int8` (sử dụng Symmetric Quantization [-127, 127]).
3. Chạy PTQ Calibration trên 150 ảnh `calibration_images/` để tính toán dải activation (Scale & Zero-point).
4. Lưu mô hình INT8 vào `models/netG_4x_quantized_int8.pth`.

**Lệnh thực thi trong Kaggle Notebook:**
```bash
!python scripts/quantize_model.py \
    --weights ./models/netG_4x_epoch5.pth.tar \
    --calib_dir ./calibration_images \
    --output ./models/netG_4x_quantized_int8.pth
```

---

### Bước 3: Đánh giá mô hình INT8 vừa Quantize
Chạy lại script `scripts/evaluate_model.py` cho mô hình INT8 để đo mức suy giảm PSNR/SSIM:

**Lệnh thực thi trong Kaggle Notebook:**
```bash
!python scripts/evaluate_model.py \
    --eval_dir ./eval_images \
    --weights ./models/netG_4x_quantized_int8.pth \
    --output_csv ./logs/quantized_int8_metrics.csv
```

---

### Phase 5: Trích xuất trọng số cho FPGA Zynq (SẮP THỰC HIỆN)
Sau khi xác nhận chất lượng mô hình INT8:
- Viết `scripts/extract_weights.py` xuất file `.txt` chứa weights, scale, zero-point của từng layer.
- Viết script `scripts/image_to_txt.py` và `scripts/txt_to_image.py` chuyển đổi định dạng dữ liệu cho AXI DMA.
