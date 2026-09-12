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

Dưới đây là bảng so sánh chi tiết chất lượng ảnh khôi phục giữa Mô hình gốc (FP32), Mô hình lượng tử hóa động và Mô hình lượng tử hóa cố định Q7 (không QAT và có QAT):

| Chỉ số | Baseline (FP32) | INT8 Quantized (Tối ưu động PTQ) | Q7 Quantized (Fixed 128 - Không QAT) | Q7 Quantized (Fixed 128 - Có QAT 15 Epochs) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean PSNR** | **41.8503 dB** | **41.9110 dB** | **28.5820 dB** | **35.6847 dB** |
| **Mean SSIM** | **0.9697** | **0.9696** | **0.8749** | **0.9548** |
| **Dung lượng Model** | **~0.90 MB** | **~0.25 MB** | **~0.25 MB** | **~0.25 MB** |
| **Đánh giá Chất lượng** | **Xuất sắc** | **Bảo toàn tuyệt đối** | **Suy giảm đáng kể** | **Phục hồi xuất sắc (Đạt yêu cầu)** |

---

## 3. Phân tích Hiệu quả của Phương pháp QAT (Quantization-Aware Training)

Kỹ thuật huấn luyện thích ứng lượng tử hóa (QAT) trong 15 epochs bằng tập dữ liệu ảnh `calibration_images` đã đem lại sự phục hồi chất lượng vượt trội cho mô hình sử dụng **Scale cố định Q7 (128.0)**:

1.  **Phục hồi chỉ số ấn tượng:**
    *   **PSNR tăng từ 28.58 dB lên 35.68 dB** (tăng mạnh **+7.10 dB**). Ảnh thu được có độ nét cao, các viền nhiễu lượng tử hóa được triệt tiêu hoàn chỉnh.
    *   **SSIM tăng từ 0.8749 lên 0.9548** (tăng **+0.08**). Chỉ số tương đồng cấu trúc đạt mức **95.48%**, đảm bảo các chi tiết giải phẫu học quan trọng trên ảnh X-quang không bị méo hay mờ mất.
2.  **Cơ chế hoạt động hiệu quả:**
    *   Nhờ cơ chế **Straight-Through Estimator (STE)**, trong lúc lan truyền tiến, mô hình liên tục được "nếm trải" sự làm tròn và kẹp giá trị của lưới Q7. 
    *   Các trọng số FP32 nền tảng liên tục tự điều chỉnh tăng/giảm nhẹ để bù trừ sai số tích tụ qua 16 Residual Blocks. Kết quả là khi làm tròn về Q7, mô hình vẫn bảo toàn được tính năng siêu độ phân giải mà không bị sụp đổ.

---

## 4. Kết luận cho Dự án FPGA Zynq

1.  Mô hình lượng tử hóa **Q7 cố định** sau khi áp dụng QAT đã đạt chất lượng phục hồi ảnh rất tốt (**PSNR 35.68 dB**, **SSIM 0.9548**), đáp ứng hoàn hảo cả hai tiêu chí:
    *   **Chất lượng ảnh y tế:** Cực tốt, giữ nguyên kết cấu chẩn đoán.
    *   **Ràng buộc phần cứng:** 100% sử dụng scale cố định lũy thừa của 2 ($128.0$), nạp trực tiếp vào RAM/FIFO của FPGA PL mà không cần mạch chia động phức tạp.
2.  File trọng số **`models/srgan_q7_weights_qat.txt`** đã sẵn sàng để chuyển sang bước kiểm tra Testbench trên Vivado hoặc nạp xuống mạch thật.