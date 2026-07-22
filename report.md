# Báo cáo Đánh giá và Thực nghiệm Lượng tử hóa (Quantization) Mô hình Swift-SRGAN

---

## 1. Mô hình Baseline (FP32)

### 1.1. Cấu trúc Mô hình Baseline (Generator)
Mạng Generator của Swift-SRGAN được thiết kế để nâng độ phân giải ảnh y tế X-quang từ $256 \times 256$ (LR) lên $1024 \times 1024$ (SR) với hệ số phóng to (upscale factor) $\times 4$. Toàn bộ trọng số sử dụng kiểu dữ liệu số thực dấu phẩy động 32-bit (`float32`).

Cấu trúc chi tiết gồm 5 thành phần chính:
1. **Initial ConvBlock:**
   - Lớp tích chập phân tách chiều sâu (SeperableConv2d) kích thước kernel $9 \times 9$, stride $1$, padding $4$.
   - Input: $3$ channels $\rightarrow$ Output: $64$ channels.
   - Hàm kích hoạt: `PReLU(num_parameters=64)`. Không sử dụng BatchNorm.
2. **16 Khối Residual Blocks (x16 Residual Blocks):**
   - Mỗi Residual Block gồm 2 ConvBlock nối tiếp nhau:
     - **Block 1:** SeperableConv2d ($3 \times 3$, padding 1) $\rightarrow$ BatchNorm2d(64) $\rightarrow$ PReLU(64).
     - **Block 2:** SeperableConv2d ($3 \times 3$, padding 1) $\rightarrow$ BatchNorm2d(64) (không dùng hàm kích hoạt).
   - Đường nối tắt (Skip Connection): $out = \text{Block 2}(x) + x$ (phép cộng element-wise).
3. **Intermediate ConvBlock:**
   - SeperableConv2d ($3 \times 3$, padding 1) $\rightarrow$ BatchNorm2d(64).
   - Cộng đường nối tắt từ lớp Initial: $out = \text{Intermediate}(x) + \text{Initial}(x)$.
4. **Upsampler Blocks (x2 Upsample Blocks):**
   - Phóng to độ phân giải $256 \times 256 \rightarrow 512 \times 512 \rightarrow 1024 \times 1024$.
   - Mỗi block gồm: SeperableConv2d ($3 \times 3$, $64 \rightarrow 256$ channels) $\rightarrow$ `PixelShuffle(scale_factor=2)` $\rightarrow$ `PReLU(64)`.
5. **Final ConvBlock:**
   - SeperableConv2d ($9 \times 9$, $64 \rightarrow 3$ channels).
   - Hàm kích hoạt đầu ra: $out = \frac{\tanh(x) + 1}{2}$ để scale giá trị pixel về khoảng $[0, 1]$.

### 1.2. Đánh giá Mô hình Baseline (FP32)
* **Tập dữ liệu kiểm thử:** 110 ảnh X-quang y tế (`eval_images/`)
* **Dung lượng file trọng số:** ~0.9 MB (`netG_4x_epoch5.pth.tar`)
* **Kết quả đo lường thực tế:**

| Chỉ số | Giá trị | Đánh giá |
| :--- | :---: | :--- |
| **Mean PSNR** | **41.8503 dB** | Mức độ nhiễu/sai số pixel cực kỳ thấp. Ảnh khôi phục xấp xỉ ảnh gốc. |
| **Mean SSIM** | **0.9697** | Độ tương đồng cấu trúc hình học (xương, mô tế bào) đạt ~97%. |

---

## 2. Mô hình sau khi Quantize (Naïve INT8 PTQ)

### 2.1. Cấu trúc Mô hình sau Quantize
Áp dụng phương pháp Lượng tử hóa tĩnh sau huấn luyện (Post-Training Static Quantization - PTQ) kiểu Naïve Per-Tensor trên toàn bộ các lớp tích chập của Generator:

1. **BatchNorm Fusion:**
   - Toàn bộ các lớp `BatchNorm2d` trong 16 Residual Blocks và Intermediate Block được nén (fuse) trực tiếp vào trọng số của lớp `Conv2d` đứng trước:
     $$W_{\text{fused}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad b_{\text{fused}} = (b - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
2. **Cấu trúc từng phần sau khi Quantize:**
   - **Mọi lớp Conv2d** (cả Depthwise $3 \times 3$, $9 \times 9$ và Pointwise $1 \times 1$) chuyển sang dạng **`QuantizedConv2d`**:
     - Trọng số (Weights): Lưu dạng số nguyên 8-bit (`int8` trong khoảng $[-127, 127]$) với 1 giá trị `weight_scale` (Per-Tensor) cho toàn bộ layer.
     - Bias: Lưu dạng số thực FP32/INT32.
     - Activation Stats: Đo dải min/max trên 150 ảnh `calibration_images/` để tính `act_in_scale`, `act_in_zero_point`, `act_out_scale`, `act_out_zero_point` (dạng `uint8` $[0, 255]$).
   - **Hàm kích hoạt (PReLU, Tanh) & PixelShuffle:** Giữ nguyên tính toán dạng số thực (Float32).

### 2.2. Đánh giá Mô hình INT8 Quantized
* **Tập dữ liệu kiểm thử:** 110 ảnh X-quang y tế (`eval_images/`)
* **Dung lượng file trọng số:** ~0.25 MB (`netG_4x_quantized_int8.pth`) — Giảm ~3.6 lần dung lượng
* **Kết quả đo lường thực tế:**

| Chỉ số | Baseline (FP32) | Naïve INT8 Quantized | Mức suy giảm |
| :--- | :---: | :---: | :---: |
| **Mean PSNR** | **41.8503 dB** | **12.0049 dB** | **-29.8454 dB** (Sụt giảm nghiêm trọng) |
| **Mean SSIM** | **0.9697** | **0.6854** | **-0.2843** (Mất cấu trúc) |

---

## 3. Nguyên nhân Sụt giảm Chất lượng Nghiêm trọng sau khi Quantize

Chỉ số PSNR tụt dốc từ **41.85 dB xuống 12.00 dB** cho thấy mô hình bị sụp đổ (Model Collapse). Nguyên nhân kỹ thuật đến từ 3 vấn đề cốt lõi:

### 3.1. Hạn chế của Per-Tensor Quantization đối với Depthwise Separable Convolution
- Kiến trúc Swift-SRGAN phụ thuộc nặng vào **Depthwise Convolution** (mỗi channel ảnh được tính bởi 1 bộ lọc kernel riêng).
- Biên độ giá trị (dynamic range) giữa các channel khác nhau có sự chênh lệch rất lớn.
- Khi quantize ngây thơ kiểu **Per-Tensor** (dùng 1 giá trị Scale duy nhất cho toàn bộ tensor 4D của layer), các channel có biên độ trọng số nhỏ bị làm tròn (rounding) hoàn toàn về $0$. Điều này làm mất đi hàng loạt kênh thông tin đặc trưng quan trọng.

### 3.2. Hiện tượng Tích tụ Sai số Lượng tử hóa qua 16 Khối Residual Blocks
- Trong 16 khối Residual Blocks, đường nối tắt Skip Connection thực hiện phép cộng trực tiếp: $out = f(x) + x$.
- Khi dải giá trị Activation bị nén thô bạo vào khoảng $[0, 255]$ (uint8) mà không có clipping/scale chính xác cho từng đường cộng, sai số lượng tử hóa (quantization error) của mỗi block liên tục bị cộng dồn và khuếch đại qua 16 tầng nối tiếp.

### 3.3. Hiện tượng Bão hòa (Saturation) tại Lớp Tanh Đầu ra
- Ở lớp cuối cùng, mô hình dùng hàm $out = \frac{\tanh(x) + 1}{2}$.
- Hàm $\tanh(x)$ chỉ hoạt động tuyến tính trong khoảng hẹp xung quanh $0$ ($[-2, 2]$). Khi các layer INT8 phía trước bị lệch Scale, giá trị $x$ đi vào hàm `tanh` bị vọt lên quá lớn (ví dụ $> 10$ hoặc $< -10$).
- Kết quả là $\tanh(x)$ bị bão hòa hoàn toàn về $+1$ hoặc $-1$, khiến ảnh đầu ra bị bão hòa điểm ảnh (cháy sáng trắng xóa hoặc đen thâu), đẩy sai số MSE lên cực cao và dồn PSNR về ngưỡng nhiễu ~12 dB.

---

## 4. Đề xuất Hướng Khắc phục cho Phần cứng FPGA Zynq

1. **Chuyển sang Per-Channel Quantization:** Tính toán Scale và Zero-point riêng biệt cho từng channel trọng số của Depthwise Conv.
2. **Bảo vệ Layer Đầu & Layer Cuối (Head & Tail Preservation):** Giữ nguyên độ chính xác cao cho lớp `Initial` và lớp `Final Conv` + `Tanh`, chỉ Quantize 16 khối **Residual Blocks** trung gian (đúng với thiết kế khối tăng tốc trên phần cứng FPGA Zynq PL).