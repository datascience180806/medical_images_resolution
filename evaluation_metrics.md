# Báo Cáo Đánh Giá Hiệu Năng Hệ Thống (System Evaluation Metrics Report)

Báo cáo này phân loại và phân tích chi tiết các chỉ số thu thập được từ quá trình chạy thực tế bo mạch **PYNQ-Z2 (FPGA Xilinx Zynq-7000 XC7Z020)**. Các chỉ số được chia làm **2 nhóm chính**: Đánh giá Chất lượng Mô hình (Model Performance) và Đánh giá Phần cứng (Hardware Performance).

---

## NHÓM 1: ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH (MODEL PERFORMANCE & IMAGE QUALITY)

Nhóm chỉ số này đánh giá khả năng khôi phục ảnh y tế từ độ phân giải thấp (LR) lên độ phân giải cao (SR) về mặt toán học và độ chính xác lâm sàng, đảm bảo giữ nguyên cấu trúc giải phẫu học của ảnh X-quang.

### 1.1. Các Chỉ Số Chất Lượng Khôi Phục Ảnh (Image Restoration Metrics)
*   **Peak Signal-to-Noise Ratio (PSNR = 43.81 dB):**
    *   *Ý nghĩa:* Đo lường tỷ lệ giữa tín hiệu hữu ích cực đại và nhiễu nền được tạo ra do quá trình nội suy/lượng tử hóa. Chỉ số **43.81 dB** là cực kỳ cao đối với ảnh y tế (ngưỡng chấp nhận được thông thường là > 30 dB), chứng minh ảnh khôi phục hầu như không có nhiễu nhân tạo.
*   **Structural Similarity Index (SSIM = 0.9880):**
    *   *Ý nghĩa:* Đánh giá độ tương đồng về mặt cấu trúc giải phẫu (độ sáng, độ tương phản và biên dạng cấu trúc xương/phổi) giữa ảnh gốc chất lượng cao và ảnh khôi phục. Chỉ số **0.9880** (gần sát mức tuyệt đối 1.0) khẳng định không có sự biến dạng cấu trúc mô học.
*   **Edge Preservation Index (EPI = 0.8119):**
    *   *Ý nghĩa:* Đo lường mức độ bảo toàn các đường biên sắc nét (edges) sử dụng tương quan gradient Laplacian. Trong ảnh X-quang, biên xương và ranh giới mô mềm là tối quan trọng để bác sĩ chẩn đoán; EPI đạt **0.8119** chứng minh biên ảnh không bị mờ nhòe.
*   **Contrast-to-Noise Ratio (CNR = 0.4600):**
    *   *Ý nghĩa:* Đo tương phản giữa vùng mô xương quan tâm (Bone ROI) và nhiễu nền xung quanh. Đảm bảo mô hình nâng cao độ nét nhưng vẫn giữ được độ tương phản tự nhiên của phim X-quang.

### 1.2. Độ Chính Xác Số Học (Arithmetic & Data Integrity)
*   **Tỷ lệ khớp bit tuyệt đối (Exact Bit Match Rate = 58.24%):**
    *   *Ý nghĩa:* Tỷ lệ phần trăm các điểm ảnh có giá trị số nguyên khớp hoàn toàn 100% giữa mô phỏng phần mềm (Float32) và tính toán phần cứng số nguyên cố định (RTL Q7).
*   **Sai số tuyệt đối trung bình (MAE = 0.9265 LSB):**
    *   *Ý nghĩa:* Sai số trung bình trên mỗi pixel chỉ là **0.9265 LSB** (nhỏ hơn sai số lượng tử hóa giới hạn 1 LSB). Điều này chứng minh thuật toán ép kiểu Q7 của chúng ta kiểm soát cực tốt sai số lan truyền, đảm bảo độ trung thực dữ liệu.
*   **Sai lệch cực đại (Maximum Deviation = 21 LSB):**
    *   *Ý nghĩa:* Chỉ xuất hiện cục bộ tại các đường biên có tần số cực cao (sự chuyển đổi đột ngột giữa đen và trắng), không ảnh hưởng đến độ mịn tổng thể của ảnh.

---

## NHÓM 2: ĐÁNH GIÁ PHẦN CỨNG (HARDWARE IMPLEMENTATION METRICS)

Nhóm chỉ số này đánh giá hiệu quả thiết kế vi mạch RTL trên tấm silicon FPGA về mặt tài nguyên, tốc độ xử lý, công suất tiêu thụ và băng thông truyền dữ liệu.

### 2.1. Mức Độ Tiêu Thụ Tài Nguyên Phần Cứng (Logic Resource Utilization)
Đo lường lượng tài nguyên logic trên chip FPGA XC7Z020 mà lõi SRCNN sử dụng:
*   **Slice LUTs (15.81%):** Sử dụng 8.410 / 53.200 bộ bảng tra cứu logic. Mức sử dụng rất thấp, giúp mạch dễ dàng tích hợp thêm các tính năng khác.
*   **Slice Registers (20.15%):** Sử dụng 21.444 / 106.400 Flip-Flop để lưu trữ trạng thái và thanh ghi dịch.
*   **Block RAM (2.14%):** Chỉ tiêu tốn 3 BRAM 36K để làm bộ đệm dòng (Line Buffers) lưu trữ tạm thời dòng ảnh khi quét tích chập.
*   **DSP48E1 Slices (5.45%):** Chỉ sử dụng 12 / 220 bộ xử lý tín hiệu số chuyên dụng (nhân-cộng phần cứng). Điều này cho thấy kiến trúc mạch nhân tích chập được tối ưu cực kỳ tốt, tiết kiệm bộ nhân DSP phần cứng chuyên dụng.

### 2.2. Phân Tích Tần Số Hoạt Động & Xung Nhịp (Timing Analysis)
*   **Tần số hoạt động thiết kế (Target Clock):** Thiết lập ở mức **100.0 MHz** (Chu kỳ 10 ns).
*   **Độ lệch Setup Time (Worst Negative Slack - WNS = +2.823 ns):** Chỉ số dương (+2.823 ns) chứng minh mạch hoàn toàn không bị lỗi vi phạm thời gian đáp ứng (Zero Timing Violations).
*   **Tần số hoạt động cực đại (Maximum Operating Frequency - F_max = 139.33 MHz):** Cho thấy mạch phần cứng có dư địa timing lên tới **28.23%**, sẵn sàng ép xung lên 130 MHz nếu muốn tăng tốc độ xử lý.

### 2.3. Công Suất Tiêu Thụ & Nhiệt Độ (Power & Thermal)
*   **Tổng công suất tiêu thụ (Total On-Chip Power = 1.477 W):** Trong đó phần lớn là lõi cứng ARM PS7 chạy hệ điều hành (1.256 W).
*   **Công suất động lõi FPGA (PL Fabric Dynamic Power = 85.0 mW):** Mức tiêu thụ điện cực kỳ nhỏ (chỉ 85 mW), chứng minh thiết kế vi mạch tích chập phân tách chiều sâu tối ưu hóa năng lượng xuất sắc so với GPU máy tính (thường tốn hàng chục đến hàng trăm Watt).
*   **Nhiệt độ chip khi hoạt động (Junction Temperature = 42.0°C):** Chip chạy mát mẻ, biên độ an toàn nhiệt còn tới 43°C (Thermal Margin).

### 2.4. Tốc Độ Xử Lý & Băng Thông Hệ Thống (System Throughput & FPS)
*   **Độ trễ xử lý 1 mảnh (Single Patch Latency = 1.952 ms):** Thời gian truyền dữ liệu qua DMA cộng với thời gian tính toán tích chập trên FPGA cho 1 mảnh $128 \times 128$ pixel.
*   **Độ trễ toàn khung ảnh (Full Frame Latency = 499.84 ms):** Tổng thời gian xử lý và tái tạo hoàn chỉnh 1 bức ảnh X-quang kích thước $1024 \times 1024$ (gồm 256 mảnh).
*   **Tốc độ xử lý khung hình (Frame Rate = 2.00 FPS):** Tốc độ tái tạo ảnh thời gian thực.
*   **Hiệu năng năng lượng (Energy Efficiency):** Đạt **23.53 FPS/W** (tính riêng trên công suất động lõi FPGA).
