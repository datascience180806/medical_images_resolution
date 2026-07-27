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
| **Q7 Quantized (Fixed Scale 128 - Không QAT)** | **28.5820 dB** | **0.8749** | **~0.25 MB** | **Suy giảm đáng kể (-13.27 dB)** |

---

## 3. Các bước thực thi trên Kaggle Notebook

### Bước 1: Pull mã nguồn mới nhất từ GitHub
```bash
%cd medical_images_resolution
!git pull origin main
```

---

### Bước 2: Huấn luyện thích nghi lượng tử hóa (QAT - Quantization-Aware Training)
Để khôi phục chỉ số của mô hình khi bắt buộc sử dụng **Scale cố định Q7 (128.0)** cho phần cứng, ta thực hiện tinh chỉnh (fine-tune) mô hình bằng kỹ thuật QAT. 

Script `scripts/train_qat.py` sẽ sử dụng Straight-Through Estimator (STE) để mô phỏng lượng tử hóa Q7 trong quá trình huấn luyện bằng chính tập dữ liệu ảnh `calibration_images` có sẵn (không cần tải 45GB dữ liệu).

**Lệnh thực thi QAT trên Kaggle (khuyên dùng GPU):**
```bash
!python scripts/train_qat.py \
    --weights ./models/netG_4x_epoch5.pth.tar \
    --train_dir ./calibration_images \
    --val_dir ./eval_images \
    --output_dir ./models \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-5
```
*Sau khi chạy xong, mô hình QAT tốt nhất sẽ được lưu tại `models/netG_4x_qat_best.pth` và tự động trích xuất file trọng số Q7 hoàn hảo tại `models/srgan_q7_weights_qat.txt`.*

---

### Bước 3: Đánh giá kiểm thử bộ trọng số Q7 sau khi chạy QAT
Chạy lệnh đánh giá để kiểm nghiệm PSNR & SSIM mới sau khi chạy QAT:

```bash
!python scripts/evaluate_q7.py \
    --eval_dir ./eval_images \
    --q7_weights ./models/srgan_q7_weights_qat.txt \
    --output_csv ./logs/q7_qat_evaluation_metrics.csv
```

---

### Bước 4: Chạy Python Host Driver cho Zynq PS (OpenCV Tiling & AXI DMA)
**Lệnh chạy mô phỏng (Simulation Mode / Kaggle / PC):**
```bash
!python scripts/zynq_host_driver.py \
    --image_path ./assets/sample_lr_input.png \
    --tile_size 64 \
    --output_path ./assets/output_zynq_sr.png \
    --sim
```
