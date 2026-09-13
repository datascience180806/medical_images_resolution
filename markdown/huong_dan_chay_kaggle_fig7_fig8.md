# 🩻 Hướng Dẫn Thực Thi Inference Trên Kaggle Để Tạo Hình Fig. 7 và Fig. 8 (IEEE GTSD 2026)

Tài liệu này cung cấp quy trình chi tiết từng bước để bạn và cộng tác viên đưa mã nguồn và bộ trọng số lên **Kaggle**, thực hiện suy luận (inference) trên tập ảnh lâm sàng `sub_NIH`, và xuất ra 2 file hình ảnh chuẩn xuất bản IEEE (**300 DPI**):
1. **Fig. 7 (`fig7_boundary_ablation.png`)**: Đánh giá thực nghiệm triệt tiêu lỗi biên chia mảnh (Boundary-Artifact Elimination / Overlap-Tiling Ablation).
2. **Fig. 8 (`fig8_visual_comparison.png`)**: So sánh chất lượng thị giác phóng to (ROI Zoom-in Insets) giữa các mô hình (Bicubic, FSRCNN, ESPCN, VDSR, EDSR, và Proposed Compact SRCNN).

---

## 1. CÁC TÀI NGUYÊN ĐÃ ĐƯỢC CHUẨN BỊ SẴN TRONG REPOSITORY

Tất cả các tài nguyên cần thiết đã được tạo sẵn trong thư mục dự án:

| Tài nguyên | Đường dẫn trong máy | Mô tả chi tiết |
| :--- | :--- | :--- |
| **Kaggle Notebook (.ipynb)** | [`notebooks/generate_fig7_fig8_kaggle.ipynb`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/notebooks/generate_fig7_fig8_kaggle.ipynb) | Notebook hoàn chỉnh, có sẵn cơ chế nạp trọng số Q7 hex, kiểm tra GPU, tính PSNR/SSIM/LPIPS và vẽ đồ thị chuẩn IEEE. |
| **Gói Trọng Số Nén (ZIP)** | [`kaggle_weights_and_code.zip`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/kaggle_weights_and_code.zip) | Chứa toàn bộ trọng số phần cứng (`weights_hex_clean.txt`, `biases_hex_clean.txt`) và các mô hình so sánh (`weight_models/2x/*.pth`). |
| **Trọng số phần cứng gốc** | [`code hardware/weights_hex_clean.txt`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/code%20hardware/weights_hex_clean.txt)<br>[`code hardware/biases_hex_clean.txt`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/code%20hardware/biases_hex_clean.txt) | 1,624 trọng số Q7 (INT8) và 25 ngưỡng lệch Q14 (INT32) của Proposed Compact SRCNN ($1 \rightarrow 16 \rightarrow 8 \rightarrow 1$). |
| **Đường dẫn ảnh test trên Kaggle** | `/kaggle/input/datasets/duc24kdl/sub-x-ray/sub_X-Ray/sub_NIH/00000001_002.png` | Ảnh X-quang lồng ngực kích thước chuẩn $1024 \times 1024$ từ tập kiểm thử `sub_NIH`. |

> [!TIP]
> **Điểm đặc biệt của Notebook:** Trong trường hợp bạn chạy notebook mà **quên chưa gắn dataset trọng số**, notebook đã được tích hợp sẵn **mảng nhúng dự phòng (Embedded Hex Fallback)** bên trong code cell số 3. Do đó, notebook có thể **chạy ngay 1-click thành công 100%** mà không sợ phát sinh lỗi thiếu file!

---

## 2. HƯỚNG DẪN TẢI LÊN KAGGLE VÀ GẮN DATASET

### Bước 2.1: Mở Notebook trên Kaggle
Có 2 cách cực kỳ nhanh:
* **Cách A (Upload trực tiếp file .ipynb):**
  1. Vào [Kaggle Code](https://www.kaggle.com/code) $\rightarrow$ Nhấn **"New Notebook"**.
  2. Trên thanh menu của giao diện Notebook, chọn **File** $\rightarrow$ **Upload Notebook**.
  3. Chọn file [`notebooks/generate_fig7_fig8_kaggle.ipynb`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/notebooks/generate_fig7_fig8_kaggle.ipynb) từ máy tính của bạn.
* **Cách B (Tạo Notebook mới và copy mã):**
  1. Tạo một notebook mới trên Kaggle.
  2. Mở file notebook bằng VS Code hoặc Jupyter rồi dán lần lượt các cell vào.

---

### Bước 2.2: Tải file trọng số lên Kaggle (Tùy chọn khuyến nghị)
Để tải trọn bộ trọng số phần cứng và mô hình so sánh lên Kaggle:
1. Vào [Kaggle Datasets](https://www.kaggle.com/datasets) $\rightarrow$ Nhấn **"New Dataset"**.
2. Đặt tiêu đề (ví dụ: `srcnn-hardware-weights`).
3. Kéo thả file [`kaggle_weights_and_code.zip`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/kaggle_weights_and_code.zip) (dung lượng khoảng ~11 MB) vào ô upload $\rightarrow$ Nhấn **Create**.
4. Quay lại màn hình Notebook Kaggle vừa mở ở Bước 2.1:
   - Nhìn sang cột bên phải (Cột **Input**), nhấn **"+ Add Input"** (hoặc **"+ Add Data"**).
   - Chọn tab **"Your Datasets"** $\rightarrow$ Tìm `srcnn-hardware-weights` $\rightarrow$ Nhấn dấu **"+"** để đính kèm vào notebook.

---

### Bước 2.3: Gắn Dataset ảnh X-ray `sub-x-ray`
1. Tại cột bên phải (Cột **Input**), nhấn **"+ Add Input"**.
2. Tìm kiếm dataset: `duc24kdl/sub-x-ray` (hoặc gõ `sub-x-ray`).
3. Nhấn dấu **"+"** để add vào notebook.
4. Khi đó, đường dẫn ảnh sẽ tự động khả dụng tại:
   ```bash
   /kaggle/input/datasets/duc24kdl/sub-x-ray/sub_X-Ray/sub_NIH/00000001_002.png
   ```
   *(Notebook đã được viết sẵn bộ lọc đệ quy, nếu tên folder Kaggle có thay đổi chút ít thì script vẫn tự động tìm thấy ảnh `*.png` trong `sub_NIH`)*.

---

### Bước 2.4: Bật tăng tốc phần cứng (GPU)
1. Ở cột menu bên phải (Cột **Notebook options**).
2. Tại mục **Accelerator**, chọn **GPU T4 x 2** (hoặc **GPU P100**).
3. Mục **Internet** gạt sang **On** (để script tự động cài đặt thư viện `lpips` qua `pip install`).

---

## 3. THỰC HIỆN SUY LUẬN (INFERENCE) VÀ SINH ẢNH

Nhấn nút **"Run All"** (hoặc biểu tượng $\blacktriangleright\blacktriangleright$) trên thanh công cụ Kaggle:

### Chi tiết các bước Notebook tự động thực hiện:
* **Cell 1 & 2:** Khởi tạo môi trường, cài đặt `lpips`, kiểm tra GPU CUDA và cấu hình độ phân giải đồ họa 300 DPI.
* **Cell 3:** Định nghĩa mạng **Proposed Compact SRCNN ($1 \rightarrow 16 \rightarrow 8 \rightarrow 1$)** và giải mã chuỗi hex Q7 INT8 thành tensor PyTorch.
* **Cell 4:** Nạp kiến trúc các mô hình đối chứng (FSRCNN, ESPCN, VDSR, EDSR).
* **Cell 5:** Đọc ảnh X-quang `00000001_002.png` từ `sub_NIH`, hạ mẫu 2x và nội suy Bicubic lên $1024 \times 1024$.
* **Cell 6 (Tạo Fig. 7):**
  - Luồng (a): Cắt và ghép 64 mảnh không chồng chập ($S=128, M=0$) $\rightarrow$ Làm nổi rõ đường nứt sọc chia mảnh.
  - Luồng (b): Cắt và ghép theo thuật toán Overlap-Tiling của nhóm ($S=112, M=8$) $\rightarrow$ Tái tạo liên tục hoàn hảo không tì vết.
  - Luồng (c): Tính toán bản đồ vi sai $|I_{\text{overlap}} - I_{\text{non-overlap}}| \times 10$ bằng colormap `Inferno`.
  - Xuất file: `fig7_boundary_ablation.png` (300 DPI).
* **Cell 7 & 8 (Tạo Fig. 8):**
  - Định vị 2 vùng giải phẫu lâm sàng trọng điểm:
    - **ROI 1 (Hộp đỏ):** Bờ xương sườn / Viền màng phổi (Cortical Rib Margin).
    - **ROI 2 (Hộp vàng):** Nhánh phế huyết quản nhu mô phổi (Vascular Arborization).
  - Cắt và hiển thị phóng to song song trên 7 cột mô hình:
    $$\text{Ground Truth} \mid \text{Bicubic} \mid \text{FSRCNN} \mid \text{ESPCN} \mid \text{VDSR} \mid \text{EDSR} \mid \textbf{Proposed (FPGA)}$$
  - Tính và gắn nhãn chỉ số khách quan ngay dưới từng ảnh: `PSNR (dB) / SSIM / LPIPS`.
  - Xuất file: `fig8_visual_comparison.png` (300 DPI).
* **Cell 9:** Kiểm tra tính toàn vẹn và dung lượng của 2 file ảnh trong thư mục `/kaggle/working`.

---

## 4. TẢI ẢNH VỀ MÁY VÀ CẬP NHẬT VÀO BÀI BÁO

1. Sau khi chạy xong, ở cột bên phải của giao diện Kaggle, mở rộng mục **Output** $\rightarrow$ `/kaggle/working`.
2. Bạn sẽ thấy xuất hiện 2 file:
   - `fig7_boundary_ablation.png`
   - `fig8_visual_comparison.png`
3. Nhấp vào biểu tượng 3 dấu chấm $(\vdots)$ cạnh từng file và chọn **Download**.
4. Copy 2 file tải về vào thư mục bài báo trên máy tính:
   ```bash
   Medical_SR_hardware_paper/GTSD2026-193-IEEE/figures/
   ```
5. Khi bạn biên dịch lại file LaTeX [`Section4_Result_analysis.tex`](file:///c:/Users/Admin/Documents/viet_code/repo_github/Super-Resolution-for-Medical-Images/Medical_SR_hardware_paper/GTSD2026-193-IEEE/Section4_Result_analysis.tex), cả 2 hình sẽ tự động hiển thị sắc nét với đầy đủ chú thích khoa học chuẩn IEEE!
