# Kế hoạch: Quantize Swift-SRGAN Generator sang INT8 cho FPGA Zynq

## 1. Mục tiêu dự án

Quantize mạng **Generator** của Swift-SRGAN từ `float32` sang `int8` để trích xuất trọng số triển khai lên sơ đồ phần cứng FPGA Zynq (PS + PL) theo kiến trúc:

```
PC (Python) → Zynq PS (ARM - AXI DMA) → Zynq PL (FPGA Accelerator) → Zynq PS → PC (Python)
```

> **Lưu ý:** Chỉ cần quantize **Generator** (mạng sinh ảnh). **Discriminator** chỉ dùng trong quá trình huấn luyện và không cần triển khai lên FPGA.
> **Môi trường thực thi:** Toàn bộ công việc thực nghiệm, đánh giá và quantize sẽ chạy trên **Kaggle Notebook / Environment**.

---

## 2. Hiện trạng dự án

### 2.1. File trọng số có sẵn
| File | Kích thước | Mô tả |
|------|-----------|-------|
| `models/netG_4x_epoch5.pth.tar` | ~0.9 MB | Trọng số Generator (float32) |
| `models/netD_4x_epoch5.pth.tar` | ~77 MB | Trọng số Discriminator (không cần dùng) |

### 2.2. Kết quả huấn luyện gốc (float32 - Baseline gốc trên full dataset)
Từ file `logs/ssrgan_5_train_results.csv`, kết quả tại **epoch 5**:

| Chỉ số | Giá trị |
|--------|---------|
| **PSNR** | **41.66 dB** |
| **SSIM** | **0.9615** |
| Generator Loss | 0.00217 |

### 2.3. Tập ảnh đã chuẩn bị (đặt ở thư mục gốc)
- **`calibration_images/`**: Chứa 150 ảnh X-quang phục vụ cho việc hiệu chỉnh Quantization (PTQ calibration).
- **`eval_images/`**: Chứa 100 ảnh X-quang phục vụ cho việc đánh giá baseline (float32) và model sau khi quantize (int8).

---

## 3. Kế hoạch thực hiện chi tiết

### Phase 1: Chuẩn bị môi trường & Repo mới (ĐÃ HOÀN THÀNH)
- [x] Đã xóa lịch sử git cũ (`.git`) và khởi tạo lại repository git mới để push sang repo GitHub mới.
- [x] Đã tinh chỉnh lại [requirements.txt](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/requirements.txt) chỉ giữ các thư viện cần thiết cho việc inference và quantize (torch, torchvision, Pillow, numpy, pandas, tqdm).

---

### Phase 2: Đánh giá Model gốc (Float32 Baseline) trên tập `eval_images`

#### Bước 2.1: Script đánh giá `scripts/evaluate_model.py` (ĐÃ TẠO)
Script [evaluate_model.py](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/scripts/evaluate_model.py) thực hiện:
- Load mô hình Generator từ `models/netG_4x_epoch5.pth.tar`
- Duyệt qua từng ảnh trong `eval_images/` (downsample từ 1024×1024 xuống 256×256 bằng Bicubic)
- Đưa qua Generator thu ảnh SR (1024×1024)
- Tính toán chỉ số **PSNR** và **SSIM** trung bình trên toàn bộ tập ảnh
- Xuất file kết quả chi tiết ra `logs/baseline_fp32_metrics.csv`

#### Bước 2.2: Lệnh chạy đánh giá trên Kaggle Notebook
```bash
python scripts/evaluate_model.py \
    --eval_dir ./eval_images \
    --weights ./models/netG_4x_epoch5.pth.tar \
    --output_csv ./logs/baseline_fp32_metrics.csv
```

---

### Phase 3: Quantize Model sang INT8

#### Bước 3.1: Viết script quantize `scripts/quantize_model.py` [SẮP THỰC HIỆN]
Sử dụng **PyTorch Post-Training Static Quantization (PTQ)**:

```
Quy trình PTQ:
1. Load model float32
2. Chèn QuantStub() và DeQuantStub() vào đầu/cuối model
3. Fuse các layer liên tiếp (Conv + BN + ReLU → 1 block)
4. Cấu hình quantization backend ('qnnpack' cho ARM/FPGA hoặc 'fbgemm' cho x86 CPU)
5. Chạy calibration trên tập ảnh 150 ảnh ('calibration_images/')
6. Chuyển đổi model sang int8
7. Lưu model quantized sang file `models/netG_4x_quantized_int8.pth`
```

---

### Phase 4: Đánh giá Model sau Quantize (INT8)

#### Bước 4.1: Đánh giá chất lượng ảnh INT8
Chạy đánh giá trên tập `eval_images/`:
```bash
python scripts/evaluate_model.py \
    --eval_dir ./eval_images \
    --weights ./models/netG_4x_quantized_int8.pth \
    --output_csv ./logs/quantized_int8_metrics.csv
```

#### Bước 4.2: So sánh Baseline (Float32) vs INT8

| Chỉ số | Float32 (Baseline) | INT8 (Quantized) | Suy giảm |
|--------|--------------------|--------------------|----------|
| PSNR (dB) | ? | ? | ? |
| SSIM | ? | ? | ? |
| Dung lượng trọng số | ~0.9 MB | ~0.25 MB | ~4x |

---

### Phase 5: Trích xuất trọng số INT8 cho FPGA

#### Bước 5.1: Viết script trích xuất `scripts/extract_weights.py` [SẮP THỰC HIỆN]
Trích xuất:
- Trọng số int8 (weight tensor)
- Scale và Zero-point của từng layer
- Bias (int32)
- Xuất các file TXT vào thư mục `weights_export/`

#### Bước 5.2: Viết script chuyển đổi ảnh ↔ TXT cho FPGA [SẮP THỰC HIỆN]
- `scripts/image_to_txt.py`: Ảnh LR (256×256) → File TXT int8 (cho AXI DMA)
- `scripts/txt_to_image.py`: File TXT từ AXI DMA → Ảnh SR (1024×1024)

---

## 4. Cấu trúc dự án hiện tại

```
Super-Resolution-for-Medical-Images/
├── calibration_images/         ← 150 ảnh X-quang cho PTQ calibration
├── eval_images/                ← 100 ảnh X-quang cho evaluation
├── models/
│   ├── netG_4x_epoch5.pth.tar  ← Model Generator float32 gốc
│   └── netD_4x_epoch5.pth.tar
├── scripts/
│   ├── evaluate_model.py       ← [MỚI] Script đánh giá PSNR/SSIM
│   ├── model_architecture.py   ← Định nghĩa mạng Generator/Discriminator
│   ├── model_metrics.py        ← Hàm tính SSIM & PSNR
│   └── ...
├── logs/
│   └── baseline_fp32_metrics.csv ← [KẾT QUẢ ĐẦU RA]
├── requirements.txt            ← Chỉ gồm thư viện cho inference
└── plan.md                     ← File kế hoạch này
```
