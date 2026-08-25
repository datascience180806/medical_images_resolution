# Danh Mục Các Chỉ Số Đánh Giá Hệ Thống (System Evaluation Metrics)

Tài liệu này định nghĩa danh mục các chỉ số cần đo lường và đánh giá khi triển khai thực nghiệm mô hình Siêu độ phân giải ảnh y tế (Swift-SRGAN/SRCNN) lên nền tảng phần cứng tăng tốc FPGA (PYNQ-Z2). Toàn bộ các chỉ số được phân loại và cấu trúc thành **2 nhóm chính**: Đánh giá Mô hình (Model Performance) và Đánh giá Phần cứng (Hardware Performance).

---

## NHÓM 1: ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH (MODEL PERFORMANCE & IMAGE QUALITY)

Nhóm chỉ số này tập trung đánh giá chất lượng phục hồi hình ảnh X-quang (độ sắc nét, độ tương đồng cấu trúc giải phẫu, cảm nhận thị giác) bằng cả phương pháp khách quan (Objective), chủ quan (Subjective) và độ chính xác số học lượng tử hóa.

### 1.1. Các Độ Đo Chất Lượng Hình Ảnh Khách Quan (Objective Image Quality Metrics)

#### A. Nhóm Độ Đo Có Ảnh Tham Chiếu (Full-Reference Metrics)
Các chỉ số này đánh giá chất lượng ảnh siêu độ phân giải (SR) bằng cách so sánh trực tiếp với ảnh gốc chất lượng cao (HR).

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **PSNR** *(Peak Signal-to-Noise Ratio)* | Đo tỷ lệ giữa công suất tín hiệu cực đại và công suất nhiễu lỗi khôi phục ở cấp độ pixel. | **Càng cao càng tốt.** Đối với ảnh y tế, chỉ số **PSNR > 30 dB** là đạt chuẩn, và **> 35 - 40 dB** là lý tưởng. |
| **SSIM** *(Structural Similarity)* | Đánh giá độ tương đồng về cấu trúc giải phẫu hình ảnh dựa trên độ tương phản, độ sáng và cấu trúc hình học. | Dải từ 0 đến 1. **Càng gần 1 càng tốt.** Trong y khoa, bắt buộc **SSIM > 0.95** (lý tưởng là **> 0.98**). |
| **MSE / RMSE** | Sai số bình phương trung bình (hoặc căn bậc hai) đo chênh lệch giá trị điểm ảnh thô giữa ảnh SR và HR. | **Càng nhỏ càng tốt** (tiệm cận về 0, đo lường sự tương quan chặt chẽ ở mức pixel). |
| **MS-SSIM** *(Multi-scale SSIM)* | Phiên bản SSIM mở rộng, đánh giá chi tiết cấu trúc ảnh trên nhiều độ phân giải (thang đo) khác nhau để phản ánh chi tiết tốt hơn. | Dải từ 0 đến 1. **Càng gần 1 càng tốt** (SSIM đa quy mô lý tưởng **> 0.96**). |
| **LPIPS** *(Learned Perceptual Similarity)* | Đo lường khoảng cách cảm nhận thị giác của con người sử dụng đặc trưng trích xuất từ mạng nơ-ron sâu (VGG/AlexNet). | **Càng nhỏ càng tốt** (dưới **0.15** là cực tốt). Phạt nặng các ảnh mờ, giúp ưu tiên các ảnh sắc nét có vân hạt tự nhiên. |
| **EPI** *(Edge Preservation Index)* | Đo khả năng bảo toàn các đường biên, ranh giới sắc nét của các cơ quan giải phẫu sử dụng tương quan gradient Laplacian. | Dải từ 0 đến 1. **Càng gần 1 càng tốt.** Chỉ số **> 0.75** thể hiện biên cạnh sắc nét, không bị mờ nhòe vùng biên. |
| **CNR** *(Contrast-to-Noise Ratio)* | Đo tỷ số tương phản trên nhiễu giữa vùng mô quan tâm (Region of Interest - ROI) và vùng nền xung quanh. | **Càng cao càng tốt.** Đảm bảo ảnh tăng độ nét nhưng vẫn giữ được độ tương phản tự nhiên của phim chụp bức xạ thực tế. |
| **IFC** *(Information Fidelity Criterion)* | Đánh giá lượng thông tin trung thực được truyền tải từ ảnh gốc đến ảnh khôi phục dựa trên lý thuyết thông tin. | **Càng cao càng tốt.** |

#### B. Nhóm Độ Đo Không Cần Ảnh Tham Chiếu (No-Reference Metrics)
Các chỉ số này đánh giá chất lượng ảnh trực tiếp mà không cần so sánh với ảnh HR gốc (rất hữu ích khi chạy dữ liệu lâm sàng thực tế).

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **NIQE** *(Naturalness Image Quality)* | Đánh giá độ tự nhiên và sự xuất hiện của các nhiễu nhân tạo/đường răng cưa trên ảnh SR dựa trên mô hình thống kê ảnh tự nhiên. | **Càng nhỏ càng tốt.** (Chỉ số NIQE lý tưởng **< 4.0** cho ảnh y tế tự nhiên). |
| **Mean** | Đo độ sáng trung bình của toàn bộ các pixel trong ảnh kết quả SR. | Phù hợp với dải sáng tiêu chuẩn của thiết bị hiển thị, tránh hiện tượng ảnh bị cháy sáng hoặc quá tối. |
| **STD** *(Standard Deviation)* | Độ lệch chuẩn của giá trị pixel, phản ánh độ tương phản và độ sắc nét tổng thể của ảnh. | **Càng cao càng tốt** (ở mức hợp lý), biểu thị ảnh có độ tương phản và độ sắc nét rõ ràng, chi tiết không bị bão hòa. |

---

### 1.2. Đánh Giá Chủ Quan (Subjective Metrics)
Đo lường mức độ hữu ích lâm sàng thực tế thông qua mắt nhìn của con người.

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **MOS** *(Mean Opinion Score)* | Điểm đánh giá chất lượng thị giác lâm sàng được chấm bởi hội đồng chuyên gia y tế (bác sĩ chẩn đoán hình ảnh). | Thang điểm từ 1 (kém) đến 5 (xuất sắc). Điểm **MOS >= 4.0** chứng minh ảnh siêu độ phân giải đáp ứng tốt nhu cầu chẩn đoán bệnh lâm sàng. |

---

### 1.3. Độ Chính Xác Số Học & Lượng Tử Hóa (Arithmetic & Quantization Integrity)
Đánh giá mức độ sai số toán học phát sinh khi chuyển đổi mô hình từ dấu phẩy động (Float32) sang số nguyên (Fixed-point Q7) chạy trên bo mạch.

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **Exact Bit Match Rate** | Tỷ lệ phần trăm các điểm ảnh có giá trị số nguyên khớp hoàn toàn 100% giữa mô phỏng phần mềm và tính toán phần cứng RTL. | **Càng cao càng tốt.** Do sai số làm tròn khi lượng tử hóa Q7, tỷ lệ này thường dao động trong khoảng **> 50% - 90%** là đạt chuẩn. |
| **MAE** *(Mean Absolute Error)* | Sai số tuyệt đối trung bình trên từng điểm ảnh giữa kết quả mô phỏng phần mềm và tính toán phần cứng (tính theo đơn vị LSB). | **Càng nhỏ càng tốt.** Bắt buộc chỉ số **MAE < 1.0 LSB** để đảm bảo sai số lượng tử hóa bị giới hạn trong phạm vi cho phép. |
| **Max Delta** *(Maximum Deviation)* | Khoảng sai lệch điểm ảnh lớn nhất xuất hiện giữa tính toán phần mềm và phần cứng. | **Càng nhỏ càng tốt.** Chỉ số này thường xuất hiện cục bộ tại các vùng biên có tần số thay đổi cực mạnh, thông thường **< 25 LSB** là chấp nhận được. |
| **Overflow / Underflow Check** | Kiểm tra lỗi tràn số vật lý khi mạch thực hiện phép cộng/nhân tích lũy ở các lớp tích chập sâu. | **Bắt buộc đạt trạng thái PASSED (0 lỗi tràn).** Mạch phải có logic bão hòa (Saturation/Clipping logic) để kẹp giá trị nếu vượt dải $[-128, 127]$. |

---

## NHÓM 2: ĐÁNH GIÁ PHẦN CỨNG (HARDWARE PERFORMANCE & EFFICIENCY)

Nhóm chỉ số này tập trung đánh giá hiệu quả thiết kế mạch số trên chip FPGA về mặt tối ưu hóa tài nguyên logic, tốc độ xử lý vật lý, độ trễ và điện năng tiêu thụ.

### 2.1. Sử Dụng Tài Nguyên Phần Cứng (Logic Resource Utilization)

| Loại Tài Nguyên (Resource) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **Slice LUTs** *(Look-Up Tables)* | Tài nguyên logic cơ bản trên chip dùng để cấu hình các hàm Boolean và phép toán logic. | **Càng thấp càng tốt.** Mức sử dụng tối ưu là **< 70%** tổng tài nguyên của chip để tránh tắc nghẽn định tuyến dây dẫn. |
| **Slice Registers / FF** *(Flip-Flops)* | Các phần tử nhớ phần cứng dùng làm thanh ghi dịch, lưu trữ trạng thái và trễ xung nhịp trong đường ống (pipeline). | **Càng thấp càng tốt.** Tương tự LUT, nên duy trì mức **< 70%** để tối ưu hóa diện tích mạch silicon. |
| **Block RAM (BRAM 36K)** | Các khối bộ nhớ RAM nội bộ chuyên dụng trên chip, dùng làm Line Buffers để đệm dữ liệu dòng quét tích chập. | **Càng thấp càng tốt.** Thường chiếm dụng **< 20%** tổng BRAM của chip để chừa tài nguyên cho các bộ nhớ đệm DMA khác. |
| **DSP48E1 Slices** | Các khối xử lý tín hiệu số phần cứng chuyên dụng thực hiện nhân-cộng song song tốc độ cao. | **Càng thấp càng tốt.** Thể hiện mạch thiết kế thông minh, tái sử dụng được tài nguyên tính toán thay vì lạm dụng DSP phần cứng. |

### 2.2. Phân Tích Timing & Xung Nhịp (Timing & Clock Frequency)

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **Worst Negative Slack (WNS)** | Khoảng thời gian dự trữ đáp ứng mạch của đường truyền tín hiệu chậm nhất (độ lệch Setup Time). | **Bắt buộc WNS > 0 ns.** Chỉ số dương chứng minh mạch thiết kế hoàn chỉnh, không có vi phạm timing ở tần số xung nhịp đích. |
| **Worst Hold Slack (WHS)** | Khoảng thời gian dự trữ tối thiểu để giữ ổn định dữ liệu tại thanh ghi (độ lệch Hold Time). | **Bắt buộc WHS > 0 ns.** Chứng minh tín hiệu không bị chạy đua (race condition) giữa các xung nhịp. |
| **F_max** *(Max Frequency)* | Tần số hoạt động cực đại về mặt vật lý mà thiết kế mạch RTL có thể đáp ứng được trên chip FPGA. | **Càng cao càng tốt.** Cho biết độ ổn định và tiềm năng ép xung tăng tốc của lõi phần cứng (thường **> 120 MHz**). |

### 2.3. Công Suất Tiêu Thụ & Nhiệt Độ (Power & Thermal)

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **PL Fabric Dynamic Power** | Điện năng tiêu thụ động sinh ra bởi quá trình chuyển mạch các cổng logic vật lý trên chip FPGA. | **Càng thấp càng tốt.** Cho biết độ tối ưu năng lượng của kiến trúc mạch (thường **< 200 mW** cho các thiết kế vi mạch di động). |
| **Junction Temperature** | Nhiệt độ tại lớp tiếp giáp bán dẫn bên trong lòng con chip khi hoạt động liên tục ở tần số cao. | **Càng thấp càng tốt.** Phải nằm dưới ngưỡng nhiệt độ giới hạn của chip (thường **< 85°C** đối với chip thương mại). |
| **Thermal Margin** | Biên độ an toàn nhiệt độ còn lại của con chip trước khi bị quá nhiệt dẫn đến hỏng hóc hoặc ngắt mạch. | **Càng cao càng tốt**, đảm bảo bo mạch hoạt động bền bỉ trong thời gian dài. |

### 2.4. Tốc Độ Xử Lý & Hiệu Năng Tính Toán (Throughput & Hardware Efficiency)

| Tên Chỉ Số (Metric) | Ý Nghĩa / Định Nghĩa | Tiêu Chuẩn Đánh Giá (Khoảng tốt) |
| :--- | :--- | :--- |
| **Single Patch Latency** | Thời gian trễ để xử lý hoàn chỉnh 1 mảnh ảnh (bao gồm truyền DMA xuống PL + chạy qua pipeline tích chập). | **Càng thấp càng tốt.** (Thời gian xử lý từng mảnh tính bằng mili-giây, ví dụ **< 5 ms**). |
| **Full Frame Latency** | Tổng thời gian trễ đầu-cuối để khôi phục và ghép nối hoàn chỉnh một bức ảnh y tế kích thước lớn ($1024 \times 1024$). | **Càng thấp càng tốt.** Đối với ứng dụng thời gian thực lâm sàng, độ trễ này nên nằm trong khoảng **< 500 ms**. |
| **Frame Processing Rate (FPS)** | Số lượng khung hình độ phân giải cao ($1024 \times 1024$) hệ thống có thể dựng lại thành công trong một giây. | **Càng cao càng tốt.** Thể hiện tốc độ xử lý thời gian thực của toàn bộ hệ thống (Host + FPGA). |
| **GOPS** *(Giga-Operations Per Second)* | Băng thông tính toán thô của mạch, đo tổng số tỷ phép toán (chủ yếu là cộng/nhân tích lũy MAC) mạch thực hiện được trong 1 giây. | **Càng cao càng tốt.** Chứng minh sức mạnh tính toán song song thực tế của phần cứng. |
| **GOPS / W** | Hiệu năng tính toán trên mỗi đơn vị điện năng tiêu thụ (Dynamic Power), thước đo chuẩn để so sánh hiệu quả năng lượng giữa các bài báo FPGA. | **Càng cao càng tốt.** Đây là thế mạnh cạnh tranh cốt lõi của FPGA so với CPU và GPU. |
| **GOPS / DSP** | Hiệu quả sử dụng tài nguyên toán học chuyên dụng, đo lường năng lực tính toán gánh trên mỗi khối DSP phần cứng. | **Càng cao càng tốt.** Thể hiện khả năng tối ưu hóa sơ đồ kiến trúc mạch để đạt hiệu suất cao với ít tài nguyên nhất. |
| **GOPS / kLUT** | Năng lực tính toán phân bổ trên mỗi 1.000 bảng tra cứu logic (LUT). | **Càng cao càng tốt.** |
| **DDR/DRAM Bandwidth** | Băng thông truy cập bộ nhớ ngoài (đọc/ghi RAM thông qua bus AXI DMA) đo bằng MB/s. | **Càng thấp càng tốt** (hoặc nằm trong giới hạn băng thông bus). Tối thiểu hóa việc truy cập bộ nhớ ngoài giúp tránh hiện tượng nghẽn cổ chai bộ nhớ (memory bottleneck). |
| **Energy Efficiency (FPS/W)** | Hiệu suất năng lượng tính bằng số khung hình xử lý được trên mỗi Watt điện năng tiêu thụ (Dynamic Power). | **Càng cao càng tốt.** |

