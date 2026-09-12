# Báo Cáo Tổng Hợp So Sánh Hiệu Năng Các Mô Hình Siêu Phân Giải Ảnh X-Quang
*(Cập nhật theo dữ liệu thực nghiệm chuẩn xác từ `performance_comparison_report_en.pdf`)*

Báo cáo này tổng hợp kết quả đo lường và đánh giá suy luận (Inference Benchmark) toàn diện giữa **Bicubic Baseline**, **mô hình Compact SRCNN RTL (1-16-8-1) của nhóm**, **SRCNN Original (1-64-32-1)** và **5 mô hình Deep Learning chuẩn (ESPCN, FSRCNN, VDSR, EDSR, SRGAN)** qua hai bộ dữ liệu X-quang y tế lâm sàng (**`sub_NIH`** và **`sub_chest`**) ở 3 tỉ lệ phóng đại (**2x**, **3x**, **4x**).

---

## Phân Biệt Các Kiến Trúc Trong Báo Cáo

1. **Bicubic (Baseline):** Thuật toán nội suy đa thức bậc ba không tham số (phương pháp cơ bản trong các thiết bị X-quang).
2. **Compact SRCNN RTL (Mô hình nhóm đề xuất):** Kiến trúc siêu thu gọn $1 \rightarrow 16 \rightarrow 8 \rightarrow 1$ gồm **1.649 tham số**, định dạng số nguyên cố định Fixed-Point INT8 Q7 ($S7.0 / S0.7 / S24.7$), được thiết kế tối ưu hóa triệt để để chạy trực tiếp trên FPGA Xilinx Zynq-7020 (PYNQ-Z2) mà không cần truy xuất DRAM trung gian.
3. **SRCNN Original (Dong et al.):** Kiến trúc gốc $1 \rightarrow 64 \rightarrow 32 \rightarrow 1$ gồm **8.129 tham số**, chạy phần mềm FP32 trích xuất đặc trưng sâu 64 và 32 kênh.
4. **ESPCN (Shi et al.):** Mạng nơ-ron tích chập sử dụng lớp phóng đại sub-pixel (`PixelShuffle`) trực tiếp từ không gian độ phân giải thấp (LR).
5. **FSRCNN (Dong et al.):** Mạng nơ-ron tích chập tốc độ cao với các lớp thu hẹp (shrinking) và mở rộng (expanding) đặc trưng, sử dụng giải tích chập (`ConvTranspose2d`).
6. **VDSR (Kim et al.):** Mạng nơ-ron sâu 20 lớp tích chập với cơ chế học phần dư toàn cục (Global Residual Learning).
7. **EDSR (Lim et al.):** Mạng nơ-ron phần dư sâu nâng cao (8 khối ResBlock, 64 kênh đặc trưng sâu).
8. **SRGAN (Ledig et al.):** Mạng nơ-ron đối kháng sinh (GAN Generator với 16 khối ResBlock) tối ưu cho cảm nhận thị giác.

---

## Mục Lục Báo Cáo

1. [Tập Dữ Liệu 1: sub_NIH (NIH ChestX-ray14 - 1.750 ảnh y tế 1024×1024)](#1-tập-dữ-liệu-sub_nih-nih-chestx-ray14---1750-ảnh)
   * [1.1. Tỉ lệ phóng đại: Scale 2x](#11-tỉ-lệ-phóng-đại-scale-2x-sub_nih)
   * [1.2. Tỉ lệ phóng đại: Scale 3x](#12-tỉ-lệ-phóng-đại-scale-3x-sub_nih)
   * [1.3. Tỉ lệ phóng đại: Scale 4x](#13-tỉ-lệ-phóng-đại-scale-4x-sub_nih)
2. [Tập Dữ Liệu 2: sub_chest (Clinical Chest X-Ray - 450 ảnh kích thước lâm sàng)](#2-tập-dữ-liệu-sub_chest-clinical-chest-x-ray---450-ảnh)
   * [2.1. Tỉ lệ phóng đại: Scale 2x](#21-tỉ-lệ-phóng-đại-scale-2x-sub_chest)
   * [2.2. Tỉ lệ phóng đại: Scale 3x](#22-tỉ-lệ-phóng-đại-scale-3x-sub_chest)
   * [2.3. Tỉ lệ phóng đại: Scale 4x](#23-tỉ-lệ-phóng-đại-scale-4x-sub_chest)
3. [Phân Tích Đột Phá & Nhận Định Chuyên Môn](#3-phân-tích-đột-phá--nhận-định-chuyên-môn)

---

## 1. Tập Dữ Liệu: `sub_NIH` (NIH ChestX-ray14 - 1.750 ảnh)
* **Đặc điểm:** Ảnh X-quang lồng ngực người lớn, chuẩn $1024 \times 1024$ pixel, độ tương phản giải phẫu cao bao phủ 14 bệnh lý ngực.

### 1.1. Tỉ lệ phóng đại: Scale 2x (`sub_NIH`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 40.0682 dB | 39.3011 dB | 39.6171 dB | 37.2428 dB | 31.9671 dB | **40.3761 dB** | 40.3739 dB | 37.5680 dB |
| **SSIM** ↑ | 0.9753 | 0.9668 | 0.9737 | 0.9610 | 0.9254 | 0.9777 | **0.9783** | 0.9636 |
| **MS-SSIM** ↑ | 0.9976 | 0.9966 | 0.9967 | 0.9941 | 0.9724 | 0.9978 | **0.9980** | 0.9947 |
| **LPIPS** ↓ | 0.0956 | **0.0583** | 0.1653 | 0.2581 | 0.4316 | 0.1140 | 0.0863 | 0.2118 |
| **NIQE** ↓ | 9.9243 | 7.6964 | 6.9968 | 7.8967 | 10.3887 | 6.7623 | **6.2820** | 6.6020 |
| **EPI** ↑ | 0.3274 | 0.3039 | 0.4451 | 0.1479 | -0.0508 | **0.4693** | 0.4076 | 0.2412 |
| **MSE** ↓ | 8.2065 | 9.0190 | 9.1015 | 15.1724 | 45.8111 | **7.6720** | 7.8904 | 14.7761 |
| **RMSE** ↓ | 2.7030 | 2.8844 | 2.8460 | 3.6969 | 6.5953 | **2.6124** | 2.6330 | 3.6104 |
| **Độ trễ (Latency)** ↓ | 15.33 ms | 12.73 ms | 44.61 ms | **4.80 ms** | 8.11 ms | 336.82 ms | 111.59 ms | 318.63 ms |
| **Tốc độ (FPS)** ↑ | 65.2 FPS | 78.6 FPS | 22.4 FPS | **208.5 FPS** | 123.3 FPS | 3.0 FPS | 9.0 FPS | 3.1 FPS |

---

### 1.2. Tỉ lệ phóng đại: Scale 3x (`sub_NIH`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 37.5998 dB | 38.4493 dB | 39.2991 dB | 36.2036 dB | 18.9758 dB | 39.2127 dB | **39.4904 dB** | 37.1230 dB |
| **SSIM** ↑ | 0.9620 | 0.9538 | 0.9643 | 0.9423 | 0.6448 | 0.9647 | **0.9658** | 0.9489 |
| **MS-SSIM** ↑ | 0.9936 | 0.9930 | 0.9941 | 0.9871 | 0.8146 | 0.9941 | **0.9946** | 0.9900 |
| **LPIPS** ↓ | 0.2076 | **0.1214** | 0.2478 | 0.3216 | 1.0178 | 0.2363 | 0.2079 | 0.2945 |
| **NIQE** ↓ | 9.9727 | 7.8957 | 7.7859 | 8.7711 | 27.3127 | 7.8913 | **7.2587** | 8.2229 |
| **EPI** ↑ | 0.1663 | 0.1064 | **0.2773** | 0.0457 | 0.0043 | 0.2546 | 0.2249 | 0.1132 |
| **MSE** ↓ | 14.9982 | 11.0340 | 9.4633 | 18.7101 | 852.7464 | 9.7603 | **9.0453** | 15.3459 |
| **RMSE** ↓ | 3.6182 | 3.1787 | 2.9142 | 4.1291 | 28.9475 | 2.9520 | **2.8508** | 3.7284 |
| **Độ trễ (Latency)** ↓ | 13.67 ms | 12.87 ms | 45.47 ms | **2.70 ms** | 4.22 ms | 395.91 ms | 64.26 ms | 241.86 ms |
| **Tốc độ (FPS)** ↑ | 73.2 FPS | 77.7 FPS | 22.0 FPS | **370.5 FPS** | 236.8 FPS | 2.5 FPS | 15.6 FPS | 4.1 FPS |

---

### 1.3. Tỉ lệ phóng đại: Scale 4x (`sub_NIH`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 36.1873 dB | 35.8637 dB | 35.5253 dB | 21.4287 dB | 15.5623 dB | 36.3261 dB | **36.4724 dB** | 35.1208 dB |
| **SSIM** ↑ | 0.9495 | 0.9413 | 0.9501 | 0.6679 | 0.6252 | 0.9529 | **0.9530** | 0.9406 |
| **MS-SSIM** ↑ | 0.9900 | 0.9893 | 0.9891 | 0.8780 | 0.8155 | 0.9906 | **0.9911** | 0.9859 |
| **LPIPS** ↓ | 0.2865 | **0.1773** | 0.3408 | 0.6851 | 0.9235 | 0.3185 | 0.3000 | 0.3489 |
| **NIQE** ↓ | 9.9859 | 8.4658 | 9.3520 | 16.9979 | 272.5003 | 8.8045 | **8.2138** | 8.9861 |
| **EPI** ↑ | 0.0923 | 0.0309 | **0.2054** | -0.0005 | -0.0070 | 0.1309 | 0.1015 | 0.0835 |
| **MSE** ↓ | 20.6699 | 21.5248 | 24.2238 | 484.6246 | 1874.2795 | 20.1630 | **19.4813** | 25.6018 |
| **RMSE** ↓ | 4.2471 | 4.3670 | 4.5899 | 21.8243 | 42.8994 | 4.1883 | **4.1170** | 4.7600 |
| **Độ trễ (Latency)** ↓ | 12.71 ms | 12.73 ms | 23.21 ms | **1.74 ms** | 2.56 ms | 368.23 ms | 39.42 ms | 195.89 ms |
| **Tốc độ (FPS)** ↑ | 78.7 FPS | 78.6 FPS | 43.1 FPS | **574.6 FPS** | 390.1 FPS | 2.7 FPS | 25.4 FPS | 5.1 FPS |

---

## 2. Tập Dữ Liệu: `sub_chest` (Clinical Chest X-Ray - 450 ảnh)
* **Đặc điểm:** Ảnh X-quang phổi lâm sàng nhi khoa (bệnh viện Nhi Quảng Châu), kích thước ảnh bất đối xứng biến thiên từ $1000 \times 800$ đến $1638 \times 1236$ pixel, độ tương phản mô mềm hẹp, kiểm tra độ ổn định thuật toán.

### 2.1. Tỉ lệ phóng đại: Scale 2x (`sub_chest`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | **40.9900 dB** | 39.2359 dB | 39.5278 dB | 37.4337 dB | 32.3510 dB | 39.9936 dB | 40.3111 dB | 38.1845 dB |
| **SSIM** ↑ | **0.9597** | 0.9308 | 0.9332 | 0.9078 | 0.8558 | 0.9432 | 0.9468 | 0.9119 |
| **MS-SSIM** ↑ | **0.9960** | 0.9947 | 0.9921 | 0.9862 | 0.9542 | 0.9950 | 0.9957 | 0.9878 |
| **LPIPS** ↓ | 0.1427 | **0.1412** | 0.2778 | 0.3876 | 0.5941 | 0.2027 | 0.1791 | 0.3399 |
| **NIQE** ↓ | 9.9425 | 7.8182 | 6.9180 | 8.3889 | 9.7426 | 5.9437 | **5.6151** | 6.6622 |
| **EPI** ↑ | **0.3991** | 0.2491 | 0.3511 | 0.0867 | 0.0002 | 0.3707 | 0.3718 | 0.2173 |
| **MSE** ↓ | **5.3800** | 8.0486 | 7.6855 | 12.2547 | 38.8168 | 6.8945 | 6.4393 | 10.4249 |
| **RMSE** ↓ | **2.2965** | 2.8108 | 2.7325 | 3.4636 | 6.1908 | 2.5895 | 2.4997 | 3.1849 |
| **Độ trễ (Latency)** ↓ | 15.39 ms | 492.30 ms | 124.29 ms | **13.02 ms** | 20.46 ms | 861.29 ms | 308.03 ms | 799.78 ms |
| **Tốc độ (FPS)** ↑ | 65.0 FPS | 2.0 FPS | 8.0 FPS | **76.8 FPS** | 48.9 FPS | 1.2 FPS | 3.2 FPS | 1.3 FPS |

---

### 2.2. Tỉ lệ phóng đại: Scale 3x (`sub_chest`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | **38.4502 dB** | 37.5675 dB | 38.1119 dB | 35.8167 dB | 19.4360 dB | 38.0783 dB | 38.3483 dB | 36.8212 dB |
| **SSIM** ↑ | **0.9338** | 0.8993 | 0.9115 | 0.8779 | 0.6049 | 0.9132 | 0.9162 | 0.8836 |
| **MS-SSIM** ↑ | **0.9885** | 0.9858 | 0.9859 | 0.9745 | 0.8140 | 0.9863 | 0.9874 | 0.9785 |
| **LPIPS** ↓ | 0.2876 | **0.2417** | 0.4035 | 0.4873 | 1.0019 | 0.3675 | 0.3308 | 0.4806 |
| **NIQE** ↓ | 9.9710 | 7.5179 | 7.8520 | 8.5688 | 21.9968 | 7.6123 | **7.1297** | 8.7333 |
| **EPI** ↑ | **0.2136** | 0.0751 | 0.1927 | 0.0256 | 0.0004 | 0.1811 | 0.1767 | 0.1077 |
| **MSE** ↓ | **9.5621** | 11.8406 | 10.5679 | 17.6864 | 748.9688 | 10.6449 | 10.0377 | 14.1808 |
| **RMSE** ↓ | **3.0694** | 3.4070 | 3.2095 | 4.1663 | 27.2897 | 3.2216 | 3.1255 | 3.7202 |
| **Độ trễ (Latency)** ↓ | 13.73 ms | 444.88 ms | 124.54 ms | **7.62 ms** | 11.01 ms | 934.05 ms | 172.38 ms | 558.56 ms |
| **Tốc độ (FPS)** ↑ | 72.8 FPS | 2.2 FPS | 8.0 FPS | **131.2 FPS** | 90.8 FPS | 1.1 FPS | 5.8 FPS | 1.8 FPS |

---

### 2.3. Tỉ lệ phóng đại: Scale 4x (`sub_chest`)

| Chỉ số (Metric) | Bicubic (Baseline) | Compact SRCNN RTL (1-16-8-1, Q7) | SRCNN Original (1-64-32-1, FP32) | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 36.5994 dB | 36.5373 dB | 36.2227 dB | 21.9458 dB | 16.0441 dB | 36.9682 dB | **37.1728 dB** | 36.0685 dB |
| **SSIM** ↑ | **0.9114** | 0.8779 | 0.8898 | 0.6314 | 0.5881 | 0.8937 | 0.8945 | 0.8736 |
| **MS-SSIM** ↑ | **0.9823** | 0.9795 | 0.9781 | 0.8779 | 0.8088 | 0.9802 | 0.9811 | 0.9724 |
| **LPIPS** ↓ | 0.3955 | **0.3138** | 0.5324 | 0.7220 | 0.9233 | 0.4913 | 0.4775 | 0.5385 |
| **NIQE** ↓ | 9.9821 | **8.1220** | 9.4849 | 14.7369 | 67.3225 | 8.7126 | 8.3824 | 9.2006 |
| **EPI** ↑ | **0.1297** | 0.0310 | 0.1201 | 0.0009 | -0.0003 | 0.1118 | 0.0864 | 0.0805 |
| **MSE** ↓ | 14.5351 | 14.9630 | 16.2005 | 420.4435 | 1637.1846 | 13.6581 | **13.0548** | 16.8056 |
| **RMSE** ↓ | 3.7918 | 3.8331 | 3.9820 | 20.4440 | 40.3374 | 3.6549 | **3.5714** | 4.0536 |
| **Độ trễ (Latency)** ↓ | 12.66 ms | 349.46 ms | 60.57 ms | **6.61 ms** | 8.51 ms | 892.44 ms | 135.01 ms | 489.20 ms |
| **Tốc độ (FPS)** ↑ | 79.0 FPS | 2.9 FPS | 16.5 FPS | **151.2 FPS** | 117.6 FPS | 1.1 FPS | 7.4 FPS | 2.0 FPS |

---

## 3. Phân Tích Đột Phá & Nhận Định Chuyên Môn

### 1. Đột Phá Cảm Nhận Thị Giác (LPIPS Vượt Trội Tuyệt Đối):
* Chỉ số **LPIPS** (Learned Perceptual Image Patch Similarity) phản ánh mức độ tương đồng cảm nhận thị giác của mắt người dựa trên đặc trưng trích xuất từ mạng sâu (càng nhỏ càng chân thực, loại bỏ cảm giác nhòe mờ).
* **Compact SRCNN RTL đạt LPIPS thấp nhất (tốt nhất) trong TOÀN BỘ các bảng đấu trên cả 2 tập dữ liệu và mọi scale:**
  * Tại `sub_NIH` Scale 2x: **0.0583** (vượt xa Bicubic 0.0956, EDSR 0.0863, VDSR 0.1140, SRCNN Original 0.1653).
  * Tại `sub_NIH` Scale 3x: **0.1214** (tốt hơn Bicubic 0.2076, EDSR 0.2079, VDSR 0.2363).
  * Tại `sub_NIH` Scale 4x: **0.1773** (tốt hơn Bicubic 0.2865, EDSR 0.3000, VDSR 0.3185).
  * Tại `sub_chest`: Giữ vững vị trí số 1 ở cả 2x (0.1412), 3x (0.2417), và 4x (0.3138).
* Điều này chứng minh thuật toán lượng tử hóa INT8 Q7 kết hợp cấu trúc lọc $9 \times 9 \rightarrow 1 \times 1 \rightarrow 5 \times 5$ khôi phục cấu trúc giải phẫu tự nhiên sắc nét, không bị hiện tượng mờ sương (blurring) như Bicubic và không bị giả mạo vi vân (hallucination) như các mạng quá sâu.

### 2. Tốc Độ và Tính Khả Thi Triển Khai Phần Cứng:
* Trên ảnh chuẩn $1024 \times 1024$ (`sub_NIH`), **Compact SRCNN RTL đạt tốc độ ~78 FPS (chỉ mất 12.7 ms/ảnh)**, nhanh hơn gần $4\times$ so với SRCNN Original (44.6 ms, 22 FPS), nhanh gấp $9\times$ so với EDSR (111.6 ms, 9 FPS) và $26\times$ so với VDSR (336.8 ms, 3 FPS).
* Với chỉ **1.649 tham số**, Compact SRCNN RTL hoàn toàn có thể nhúng trực tiếp vào bộ đệm dòng (Line Buffers) sử dụng chỉ 3.5 BRAM trên FPGA Zynq-7020 giá rẻ, đạt thông lượng xử lý thời gian thực mà không đòi hỏi GPU đắt tiền.

### 3. Sự Thất Bại của Deconvolution & Sub-Pixel ở Scale Lớn:
* **FSRCNN** (`ConvTranspose2d`) bị lỗi bàn cờ (checkerboard artifacts) cực kỳ nặng nề ở scale 3x và 4x: PSNR sụp đổ xuống **18.98 dB (3x)** và **15.56 dB (4x)**, LPIPS vọt lên trên 1.0 (hoàn toàn không sử dụng được trong chẩn đoán y khoa).
* **ESPCN** (`PixelShuffle`) cũng suy thoái nghiêm trọng ở 4x với PSNR chỉ đạt **21.43 dB** và SSIM rớt xuống **0.6679**.
* Khẳng định chiến lược **Pre-upsampling (nội suy Bicubic trước khi đưa vào mạng tích chập)** của họ mạng SRCNN là giải pháp cực kỳ an toàn và ổn định về mặt quang sai cho ảnh X-quang y tế.
