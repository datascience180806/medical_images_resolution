# Báo cáo Đánh giá và Thực nghiệm Lượng tử hóa (Quantization) Mô hình Swift-SRGAN

---

## 1. Mô hình Baseline (FP32)

### 1.1. Cấu trúc Mô hình Baseline (Generator)
Mạng Generator của Swift-SRGAN được thiết kế để nâng độ phân giải ảnh y tế X-quang từ $256 \times 256$ (LR) lên $1024 \times 1024$ (SR) với hệ số phóng to (upscale factor) $\times 4$. Toàn bộ trọng số sử dụng kiểu dữ liệu số thực dấu phẩy động 32-bit (`float32`).

### 1.2. Đánh giá Mô hình Baseline (FP32)
* **Tập dữ liệu kiểm thử:** 110 ảnh X-quang y tế (`eval_images/`)
* **Dung lượng file trọng số:** ~0.9 MB (`netG_4x_epoch5.pth.tar`)
* **Kết quả đo lường thực tế:**

| Chỉ số | Giá trị | Đánh giá |
| :--- | :---: | :--- |
| **Mean PSNR** | **41.8503 dB** | Mức độ nhiễu/sai số pixel cực kỳ thấp. Ảnh khôi phục xấp xỉ ảnh gốc. |
| **Mean SSIM** | **0.9697** | Độ tương đồng cấu trúc hình học (xương, mô tế bào) đạt ~97%. |

---

## 2. Kết quả Thực nghiệm và So sánh các Phương pháp Lượng tử hóa

Dưới đây là bảng so sánh chi tiết chất lượng ảnh khôi phục và dung lượng lưu trữ giữa Mô hình gốc (FP32) và 2 thuật toán lượng tử hóa:

| Chỉ số | Baseline (FP32) | INT8 Quantized (Optimized PTQ) | Q7 Quantized (Fixed Scale 128) |
| :--- | :---: | :---: | :---: |
| **Mean PSNR** | **41.8503 dB** | **41.9110 dB** | **28.5820 dB** |
| **Mean SSIM** | **0.9697** | **0.9696** | **0.8749** |
| **Dung lượng Model** | **~0.90 MB** | **~0.25 MB** | **~0.25 MB** |
| **Đánh giá Chất lượng** | **Xuất sắc** | **Bảo toàn tuyệt đối (Đạt yêu cầu)** | **Suy giảm đáng kể (Hạn chế)** |

---

## 3. Phân tích So sánh chất lượng mô hình Q7 Fixed-Point

So với Mô hình gốc (FP32) và mô hình INT8 tối ưu hóa động, **bộ trọng số Q7 cố định bị sụt giảm chất lượng khá lớn**:
*   **PSNR giảm từ 41.85 dB xuống 28.58 dB** (sụt giảm **-13.27 dB**).
*   **SSIM giảm từ 0.9697 xuống 0.8749** (giảm **-0.0948**).

### Nguyên nhân gây ra sự sụt giảm ở Q7 cố định:
1.  **Hệ số nhân cố định (Scale = 128.0) cho toàn bộ mô hình:**
    Khác với thuật toán tối ưu động tự động tính toán scale tối ưu cho từng channel, Q7 nhân cố định toàn bộ trọng số với $128.0$.
    *   Các trọng số quá nhỏ (ví dụ nằm trong khoảng $[-0.003, 0.003]$) sau khi nhân với 128 chỉ đạt giá trị khoảng $[-0.38, 0.38]$, khi làm tròn số nguyên (`round`) sẽ **bị triệt tiêu hoàn toàn về 0**. Điều này làm biến mất nhiều đặc trưng chi tiết nhỏ của ảnh.
    *   Các trọng số lớn ngoài khoảng $[-1.0, 1.0]$ bị kẹp cứng (`clamp`) ở $[-128, 127]$, làm méo phân phối trọng số gốc.
2.  **Độ nhạy cảm của ảnh y tế:**
    Với ảnh X-quang phổi, mức PSNR **28.58 dB** và SSIM **0.87** có nghĩa là cấu trúc thô vẫn được giữ lại nhưng **các chi tiết mô nhỏ, cạnh xương và tổn thương li ti đã bị mờ đi rõ rệt**.

---

## 4. Giải pháp cải thiện cho quy trình phần cứng FPGA

Nếu mạch phần cứng FPGA bắt buộc phải sử dụng định dạng Q7 cố định (để thiết kế mạch ALU đơn giản, không cần bộ chia/nhân động):
1.  **Áp dụng QAT (Quantization-Aware Training):** Thực hiện huấn luyện tinh chỉnh (Fine-tune) mô hình trực tiếp với hàm giả lập Q7 trong quá trình train để mô hình thích nghi trước với hệ số nhân 128.
2.  **Sử dụng Qx.y động giữa các Layer:** Nếu Vivado hỗ trợ, thay vì dùng Q7 (7 bit thập phân) cho tất cả các layer, ta có thể dùng Q5.3 cho layer có trọng số lớn hoặc Q9 cho các layer có trọng số nhỏ để tránh mất mát thông tin.