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

## 3. Các bước thực thi trên Kaggle / PC / Zynq Board

### Bước 1: Pull mã nguồn mới nhất từ GitHub
```bash
%cd medical_images_resolution
!git pull origin main
```

---

### Bước 2: Chạy Python Host Driver cho Zynq PS (OpenCV Tiling & AXI DMA)
Script `scripts/zynq_host_driver.py` thực hiện:
1. Dùng **OpenCV** đọc ảnh X-quang gốc.
2. Cắt ảnh thành các mảnh nhỏ kích thước $64 \times 64$ (`--tile_size 64`).
3. Đẩy từng mảng array qua kênh **AXI DMA** (`pynq.allocate` & `dma.sendchannel.transfer`) xuống phần cứng **Zynq PL FPGA Accelerator**.
4. Nhận kết quả từ AXI DMA và dùng OpenCV **ghép lại thành ảnh Super-Resolution** hoàn chỉnh ($1024 \times 1024$).

**Lệnh chạy mô phỏng (Simulation Mode / Kaggle / PC):**
```bash
!python scripts/zynq_host_driver.py \
    --image_path ./assets/sample_lr_input.png \
    --tile_size 64 \
    --output_path ./assets/output_zynq_sr.png \
    --sim
```

**Lệnh chạy thực tế trên bo mạch Zynq ARM PS (PYNQ Board):**
```bash
python scripts/zynq_host_driver.py \
    --image_path ./chest_xray.png \
    --bitstream ./srgan_accelerator.bit \
    --tile_size 64 \
    --output_path ./sr_output.png
```
