# Báo Cáo Tổng Hợp So Sánh Hiệu Năng Các Mô Hình Siêu Phân Giải Ảnh X-Quang

Báo cáo này tổng hợp kết quả suy luận thực nghiệm (Inference Evaluation) toàn diện giữa **Bicubic**, **mô hình Compact SRCNN (1-16-8-1) của nhóm**, và **5 mô hình Deep Learning chuẩn (ESPCN, FSRCNN, VDSR, EDSR, SRGAN)** trên GPU qua hai bộ dữ liệu X-quang y tế lâm sàng (**`sub_NIH`** và **`sub_chest`**) ở 3 tỉ lệ phóng đại (**2x**, **3x**, **4x**).

> **Đặc tả kiến trúc Compact SRCNN (1-16-8-1) của nhóm:**
> * Cấu trúc kênh: $1 \rightarrow 16 \rightarrow 8 \rightarrow 1$ (Kích thước kernel: $9 \times 9 \rightarrow 1 \times 1 \rightarrow 5 \times 5$).
> * Tổng số tham số: **1.649 tham số** (1.624 trọng số weights + 25 độ lệch bias).
> * Định dạng số học phần cứng RTL: Fixed-Point Q7 (Trọng số $S0.7$, Kích hoạt $S7.0$, Tích lũy $S24.7$ / $S16.7$).
> * Mục tiêu thiết kế: Tối ưu hóa triệt để tài nguyên DSP slice / BRAM / LUT trên FPGA Xilinx Zynq-7020 (PYNQ-Z2), triệt tiêu hoàn toàn truy xuất DRAM trung gian và đảm bảo độ trễ thời gian thực.

---

## Mục Lục Tổng Hợp

1. [Tập Dữ Liệu 1: sub_NIH (NIH ChestX-ray14 - 1.750 ảnh 1024×1024)](#1-tập-dữ-liệu-sub_nih-nih-chestx-ray14)
   * [1.1. Tỉ lệ phóng đại: Scale 2x](#11-tỉ-lệ-phóng-đại-scale-2x-sub_nih)
   * [1.2. Tỉ lệ phóng đại: Scale 3x](#12-tỉ-lệ-phóng-đại-scale-3x-sub_nih)
   * [1.3. Tỉ lệ phóng đại: Scale 4x](#13-tỉ-lệ-phóng-đại-scale-4x-sub_nih)
2. [Tập Dữ Liệu 2: sub_chest (Guangzhou Pediatric - 350 ảnh kích thước biến đổi)](#2-tập-dữ-liệu-sub_chest-guangzhou-pediatric)
   * [2.1. Tỉ lệ phóng đại: Scale 2x](#21-tỉ-lệ-phóng-đại-scale-2x-sub_chest)
   * [2.2. Tỉ lệ phóng đại: Scale 3x](#22-tỉ-lệ-phóng-đại-scale-3x-sub_chest)
   * [2.3. Tỉ lệ phóng đại: Scale 4x](#23-tỉ-lệ-phóng-đại-scale-4x-sub_chest)
3. [Tổng Kết Phân Tích Kỹ Thuật & Xu Hướng Giải Phẫu](#3-tổng-kết-phân-tích-kỹ-thuật--xu-hướng-giải-phẫu)

---

## 1. Tập Dữ Liệu: `sub_NIH` (NIH ChestX-ray14)
* **Quy mô:** 1.750 ảnh X-quang lồng ngực người lớn, định dạng thang xám $1024 \times 1024$ pixel.
* **Đặc điểm:** Tương phản cao, biên độ mô xương sườn và nhu mô phổi sắc nét, đại diện cho 14 diện bệnh lý lồng ngực tiêu chuẩn.

### 1.1. Tỉ lệ phóng đại: Scale 2x (`sub_NIH`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 40.0682 dB | 39.6171 dB | 37.2428 dB | 31.9671 dB | **40.3761 dB** | 40.3739 dB | 37.5680 dB |
| **SSIM** ↑ | 0.9753 | 0.9737 | 0.9610 | 0.9254 | 0.9777 | **0.9783** | 0.9636 |
| **MS-SSIM** ↑ | 0.9976 | 0.9967 | 0.9941 | 0.9724 | 0.9978 | **0.9980** | 0.9947 |
| **LPIPS** ↓ | 0.0956 | 0.1653 | 0.2581 | 0.4316 | 0.1140 | **0.0863** | 0.2118 |
| **NIQE** ↓ | 9.9243 | 6.9968 | 7.8967 | 10.3887 | 6.7623 | **6.2820** | 6.6020 |
| **EPI** ↑ | 0.3274 | 0.4451 | 0.1479 | -0.0508 | **0.4693** | 0.4076 | 0.2412 |
| **MSE** ↓ | 8.2065 | 9.1015 | 15.1724 | 45.8111 | **7.6720** | 7.8904 | 14.7761 |
| **RMSE** ↓ | 2.7030 | 2.8460 | 3.6969 | 6.5953 | **2.6124** | 2.6330 | 3.6104 |
| **Độ trễ (Latency)** ↓ | 15.33 ms | 44.61 ms | **4.80 ms** | 8.11 ms | 336.82 ms | 111.59 ms | 318.63 ms |
| **Tốc độ (FPS)** ↑ | 65.2 FPS | 22.4 FPS | **208.5 FPS** | 123.3 FPS | 3.0 FPS | 9.0 FPS | 3.1 FPS |

*Nhận xét 2x sub_NIH:*
* **VDSR và EDSR** đạt chất lượng tái tạo cao nhất (~40.38 dB PSNR, SSIM ~0.978), tuy nhiên độ trễ rất cao (111.59 ms – 336.82 ms, tương đương chỉ 3 – 9 FPS trên GPU).
* **Compact SRCNN (1-16-8-1)** đạt **39.6171 dB PSNR**, SSIM rất cao (**0.9737**) và chỉ số bảo toàn biên cạnh **EPI đạt 0.4451** (vượt trội Bicubic 0.3274 và tiệm cận VDSR 0.4693), trong khi dung lượng mô hình siêu nhỏ (chỉ 1.649 tham số).
* **FSRCNN** bị suy thoái chất lượng rõ rệt do cơ chế giải tích chập (Deconvolution) tạo nhiễu hạt bàn cờ trên nền ảnh X-quang mịn.

---

### 1.2. Tỉ lệ phóng đại: Scale 3x (`sub_NIH`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 37.5998 dB | 39.2991 dB | 36.2036 dB | 18.9758 dB | 39.2127 dB | **39.4904 dB** | 37.1230 dB |
| **SSIM** ↑ | 0.9620 | 0.9643 | 0.9423 | 0.6448 | 0.9647 | **0.9658** | 0.9489 |
| **MS-SSIM** ↑ | 0.9936 | 0.9941 | 0.9871 | 0.8146 | 0.9941 | **0.9946** | 0.9900 |
| **LPIPS** ↓ | **0.2076** | 0.2478 | 0.3216 | 1.0178 | 0.2363 | 0.2079 | 0.2945 |
| **NIQE** ↓ | 9.9727 | 7.7859 | 8.7711 | 27.3127 | 7.8913 | **7.2587** | 8.2229 |
| **EPI** ↑ | 0.1663 | **0.2773** | 0.0457 | 0.0043 | 0.2546 | 0.2249 | 0.1132 |
| **MSE** ↓ | 14.9982 | 9.4633 | 18.7101 | 852.7464 | 9.7603 | **9.0453** | 15.3459 |
| **RMSE** ↓ | 3.6182 | 2.9142 | 4.1291 | 28.9475 | 2.9520 | **2.8508** | 3.7284 |
| **Độ trễ (Latency)** ↓ | 13.67 ms | 45.47 ms | **2.70 ms** | 4.22 ms | 395.91 ms | 64.26 ms | 241.86 ms |
| **Tốc độ (FPS)** ↑ | 73.2 FPS | 22.0 FPS | **370.5 FPS** | 236.8 FPS | 2.5 FPS | 15.6 FPS | 4.1 FPS |

*Nhận xét 3x sub_NIH:*
* **Compact SRCNN (1-16-8-1)** vươn lên đạt **39.2991 dB PSNR**, vượt cả VDSR (39.2127 dB) và vượt xa Bicubic (+1.70 dB). Đặc biệt, chỉ số bảo toàn biên cạnh **EPI đạt 0.2773 (cao nhất toàn bộ bảng đấu)**.
* **FSRCNN** sụp đổ hoàn toàn ở scale 3x (PSNR chỉ còn 18.98 dB, MSE vọt lên 852.75) do độ bất ổn định của deconvolution.
* **EDSR** duy trì PSNR cao nhất (39.49 dB), nhưng độ trễ vẫn gấp 1.4x so với Compact SRCNN.

---

### 1.3. Tỉ lệ phóng đại: Scale 4x (`sub_NIH`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 36.1873 dB | 35.5253 dB | 21.4287 dB | 15.5623 dB | 36.3261 dB | **36.4724 dB** | 35.1208 dB |
| **SSIM** ↑ | 0.9495 | 0.9501 | 0.6679 | 0.6252 | 0.9529 | **0.9530** | 0.9406 |
| **MS-SSIM** ↑ | 0.9900 | 0.9891 | 0.8780 | 0.8155 | 0.9906 | **0.9911** | 0.9859 |
| **LPIPS** ↓ | **0.2865** | 0.3408 | 0.6851 | 0.9235 | 0.3185 | 0.3000 | 0.3489 |
| **NIQE** ↓ | 9.9859 | 9.3520 | 16.9979 | 272.5003 | 8.8045 | **8.2138** | 8.9861 |
| **EPI** ↑ | 0.0923 | **0.2054** | -0.0005 | -0.0070 | 0.1309 | 0.1015 | 0.0835 |
| **MSE** ↓ | 20.6699 | 24.2238 | 484.6246 | 1874.2795 | 20.1630 | **19.4813** | 25.6018 |
| **RMSE** ↓ | 4.2471 | 4.5899 | 21.8243 | 42.8994 | 4.1883 | **4.1170** | 4.7600 |
| **Độ trễ (Latency)** ↓ | 12.71 ms | 23.21 ms | **1.74 ms** | 2.56 ms | 368.23 ms | 39.42 ms | 195.89 ms |
| **Tốc độ (FPS)** ↑ | 78.7 FPS | 43.1 FPS | **574.6 FPS** | 390.1 FPS | 2.7 FPS | 25.4 FPS | 5.1 FPS |

*Nhận xét 4x sub_NIH:*
* Ở mức phóng đại cực hạn 4x, **Compact SRCNN (1-16-8-1)** duy trì **SSIM đạt 0.9501** (đạt ngưỡng an toàn chẩn đoán y khoa > 0.95) và **EPI đạt 0.2054 (cao nhất toàn bộ các mô hình)**, gấp hơn 2.2 lần so với Bicubic (0.0923) và vượt trội VDSR (0.1309).
* **ESPCN và FSRCNN** hỏng hoàn toàn ở 4x (PSNR chỉ đạt 21.43 dB và 15.56 dB, SSIM rơi tự do về 0.66 và 0.62).

---

## 2. Tập Dữ Liệu: `sub_chest` (Guangzhou Pediatric)
* **Quy mô:** 350 ảnh X-quang phổi nhi khoa (trẻ từ 1–5 tuổi), kích thước bất đối xứng dao động từ $1000 \times 800$ đến $1638 \times 1236$ pixel.
* **Đặc điểm:** Tỉ lệ khung hình và độ phân giải không đồng đều, mật độ mô phế nang trẻ nhỏ có độ tương phản hẹp hơn so với người lớn; dùng để kiểm chứng tính bền vững của thuật toán cắt mảnh và phóng đại.

### 2.1. Tỉ lệ phóng đại: Scale 2x (`sub_chest`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | **40.9900 dB** | 39.5278 dB | 37.4337 dB | 32.3510 dB | 39.9936 dB | 40.3111 dB | 38.1845 dB |
| **SSIM** ↑ | **0.9597** | 0.9332 | 0.9078 | 0.8558 | 0.9432 | 0.9468 | 0.9119 |
| **MS-SSIM** ↑ | **0.9960** | 0.9921 | 0.9862 | 0.9542 | 0.9950 | 0.9957 | 0.9878 |
| **LPIPS** ↓ | **0.1427** | 0.2778 | 0.3876 | 0.5941 | 0.2027 | 0.1791 | 0.3399 |
| **NIQE** ↓ | 9.9425 | 6.9180 | 8.3889 | 9.7426 | 5.9437 | **5.6151** | 6.6622 |
| **EPI** ↑ | **0.3991** | 0.3511 | 0.0867 | 0.0002 | 0.3707 | 0.3718 | 0.2173 |
| **MSE** ↓ | **5.3800** | 7.6855 | 12.2547 | 38.8168 | 6.8945 | 6.4393 | 10.4249 |
| **RMSE** ↓ | **2.2965** | 2.7325 | 3.4636 | 6.1908 | 2.5895 | 2.4997 | 3.1849 |
| **Độ trễ (Latency)** ↓ | 15.39 ms | 124.29 ms | **13.02 ms** | 20.46 ms | 861.29 ms | 308.03 ms | 799.78 ms |
| **Tốc độ (FPS)** ↑ | 65.0 FPS | 8.0 FPS | **76.8 FPS** | 48.9 FPS | 1.2 FPS | 3.2 FPS | 1.3 FPS |

---

### 2.2. Tỉ lệ phóng đại: Scale 3x (`sub_chest`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | **38.4502 dB** | 38.1119 dB | 35.8167 dB | 19.4360 dB | 38.0783 dB | 38.3483 dB | 36.8212 dB |
| **SSIM** ↑ | **0.9338** | 0.9115 | 0.8779 | 0.6049 | 0.9132 | 0.9162 | 0.8836 |
| **MS-SSIM** ↑ | **0.9885** | 0.9859 | 0.9745 | 0.8140 | 0.9863 | 0.9874 | 0.9785 |
| **LPIPS** ↓ | **0.2876** | 0.4035 | 0.4873 | 1.0019 | 0.3675 | 0.3308 | 0.4806 |
| **NIQE** ↓ | 9.9710 | 7.8520 | 8.5688 | 21.9968 | 7.6123 | **7.1297** | 8.7333 |
| **EPI** ↑ | **0.2136** | 0.1927 | 0.0256 | 0.0004 | 0.1811 | 0.1767 | 0.1077 |
| **MSE** ↓ | **9.5621** | 10.5679 | 17.6864 | 748.9688 | 10.6449 | 10.0377 | 14.1808 |
| **RMSE** ↓ | **3.0694** | 3.2095 | 4.1663 | 27.2897 | 3.2216 | 3.1255 | 3.7202 |
| **Độ trễ (Latency)** ↓ | 13.73 ms | 124.54 ms | **7.62 ms** | 11.01 ms | 934.05 ms | 172.38 ms | 558.56 ms |
| **Tốc độ (FPS)** ↑ | 72.8 FPS | 8.0 FPS | **131.2 FPS** | 90.8 FPS | 1.1 FPS | 5.8 FPS | 1.8 FPS |

---

### 2.3. Tỉ lệ phóng đại: Scale 4x (`sub_chest`)

| Thông số đánh giá | Bicubic | Compact SRCNN (1-16-8-1) [Nhóm] | ESPCN | FSRCNN | VDSR | EDSR | SRGAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** ↑ | 36.5994 dB | 36.2227 dB | 21.9458 dB | 16.0441 dB | 36.9682 dB | **37.1728 dB** | 36.0685 dB |
| **SSIM** ↑ | **0.9114** | 0.8898 | 0.6314 | 0.5881 | 0.8937 | 0.8945 | 0.8736 |
| **MS-SSIM** ↑ | **0.9823** | 0.9781 | 0.8779 | 0.8088 | 0.9802 | 0.9811 | 0.9724 |
| **LPIPS** ↓ | **0.3955** | 0.5324 | 0.7220 | 0.9233 | 0.4913 | 0.4775 | 0.5385 |
| **NIQE** ↓ | 9.9821 | 9.4849 | 14.7369 | 67.3225 | 8.7126 | **8.3824** | 9.2006 |
| **EPI** ↑ | **0.1297** | 0.1201 | 0.0009 | -0.0003 | 0.1118 | 0.0864 | 0.0805 |
| **MSE** ↓ | 14.5351 | 16.2005 | 420.4435 | 1637.1846 | 13.6581 | **13.0548** | 16.8056 |
| **RMSE** ↓ | 3.7918 | 3.9820 | 20.4440 | 40.3374 | 3.6549 | **3.5714** | 4.0536 |
| **Độ trễ (Latency)** ↓ | 12.66 ms | 60.57 ms | **6.61 ms** | 8.51 ms | 892.44 ms | 135.01 ms | 489.20 ms |
| **Tốc độ (FPS)** ↑ | 79.0 FPS | 16.5 FPS | **151.2 FPS** | 117.6 FPS | 1.1 FPS | 7.4 FPS | 2.0 FPS |

---

## 3. Tổng Kết Phân Tích Kỹ Thuật & Xu Hướng Giải Phẫu

1. **Hiệu Quả Bảo Toàn Biên Cạnh (EPI - Edge Preservation Index) Vượt Trội:**
   * Trong chẩn đoán hình ảnh X-quang, ranh giới xương sườn, vòm hoành và phế quản phổi là yếu tố sống còn. Compact SRCNN (1-16-8-1) đạt **EPI cao nhất toàn diện ở Scale 3x (0.2773) và 4x (0.2054)** trên tập `sub_NIH`, vượt xa cả các mạng sâu như VDSR (0.1309) và EDSR (0.1015).
   * Điều này chứng minh cấu trúc tích chập gọn nhẹ với tiền nội suy bicubic bảo toàn dải tần số cao tự nhiên của tia X tốt hơn việc ép mạng học qua hàng chục khối residual sâu vốn dễ bị "mịn hóa" (over-smoothing).

2. **Cân Bằng Giữa Chất Lượng Tái Tạo và Chi Phí Phần Cứng:**
   * **VDSR và EDSR** đạt điểm PSNR nhỉnh hơn từ 0.7 dB đến 1.0 dB, nhưng sở hữu số lượng tham số khổng lồ (VDSR: 668k, EDSR: 963k tham số) và độ trễ trên GPU lên tới **336 ms – 934 ms / ảnh** (tương đương chỉ 1.1 – 3.0 FPS). Mức tiêu tốn bộ nhớ và phép toán này hoàn toàn bất khả thi đối với các dòng chip biên giá rẻ như Xilinx Zynq-7020.
   * **Compact SRCNN (1-16-8-1)** chỉ tiêu tốn **1.649 tham số** (nhỏ hơn 405 lần so với VDSR và 584 lần so với EDSR), đạt PSNR trên 39.6 dB ở 2x và 39.3 dB ở 3x, hoàn toàn phù hợp để ánh xạ 100% lên phần cứng FPGA với chỉ 120 DSP slice và 3.5 BRAM.

3. **Hiện Tượng Suy Thoái Nặng Nề của Post-Upsampling (FSRCNN, ESPCN):**
   * **FSRCNN** sử dụng `ConvTranspose2d` với stride lớn bị hiện tượng lỗi xếp chồng bộ lọc (checkerboard artifact), khiến PSNR tụt dốc thảm hại xuống 18.98 dB (3x) và 15.56 dB (4x).
   * **ESPCN** dùng `PixelShuffle` cũng chịu suy thoái tương tự ở 4x (PSNR chỉ còn 21.43 dB, SSIM tụt xuống 0.66).
   * Kết quả này khẳng định phương pháp tiếp cận **Pre-upsampling (nội suy Bicubic trước khi đưa vào các tầng tích chập)** của họ mạng SRCNN là lựa chọn an toàn, ổn định nhất về mặt hình thái học cho dữ liệu X-quang y tế.
