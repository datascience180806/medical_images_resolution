TIÊU ĐỀ: HƯỚNG DẪN NHIỆM VỤ 2 - HOÀN THIỆN PYTHON HOST (PYNQ & DMA)

Gửi ông (phụ trách Code Host giao tiếp phần cứng),

Nhiệm vụ của ông là viết kịch bản Python đóng vai trò "cầu nối" giữa thuật toán cắt/ghép ảnh X-quang và lõi AI phần cứng (FPGA) mà tui đã đúc xong. 

LƯU Ý CỰC KỲ QUAN TRỌNG: 
- Hàm cắt/ghép ảnh (OpenCV/Numpy) ông có thể test độc lập trên laptop.
- Nhưng đoạn code đẩy dữ liệu (DMA) BẮT BUỘC phải chạy trên board PYNQ. Tui đã chuẩn bị sẵn 2 file "srcnn_accel.bit" và "srcnn_accel.hwh" cho ông, khi nào lên lab cắm board thì copy 2 file này vào cùng thư mục code nhé.

1. Yêu cầu luồng dữ liệu (Workflow)
- Bước 1: Dùng OpenCV đọc ảnh X-quang gốc (Grayscale).
- Bước 2: Cắt bức ảnh to thành nhiều mảnh nhỏ kích thước cố định (Ví dụ: 64x64 pixel) vì phần cứng tui fix cứng size này rồi.
- Bước 3: Đẩy từng mảnh 64x64 (dạng Numpy array 8-bit) qua kênh DMA xuống mạch phần cứng.
- Bước 4: Nhận mảng dữ liệu kết quả từ mạch trả về qua DMA (lúc này ảnh đã được phần cứng phóng to/làm nét).
- Bước 5: Ghép các mảnh kết quả lại thành một bức ảnh X-quang siêu phân giải hoàn chỉnh và xuất ra.

2. Code Python tham khảo (Jupyter Notebook trên PYNQ):
Ông dùng bộ khung này để phát triển tiếp hàm cắt ghép nhé:

import cv2
import numpy as np
from pynq import Overlay, allocate

# 1. KHỞI TẠO PHẦN CỨNG (Chỉ chạy được trên board PYNQ)
# Đảm bảo 2 file .bit và .hwh nằm cùng thư mục
print("Đang nạp file bitstream xuống FPGA...")
overlay = Overlay("srcnn_accel.bit")
dma = overlay.axi_dma_0 # Tên kênh DMA (tùy cấu hình tui set)

# 2. CHUẨN BỊ BỘ NHỚ ĐỆM (Buffer) CHO DMA
# Cấp phát vùng nhớ liên tục trong RAM để phần cứng có thể đọc/ghi trực tiếp
# Giả sử đầu vào là mảnh 64x64, đầu ra phóng to x2 là 128x128
in_buffer = allocate(shape=(64 * 64,), dtype=np.uint8)
out_buffer = allocate(shape=(128 * 128,), dtype=np.uint8)

# 3. HÀM XỬ LÝ LÕI TỪNG MẢNH ẢNH
def process_patch_on_fpga(patch_64x64):
    # Đổ mảng Numpy vào buffer đầu vào của DMA
    in_buffer[:] = patch_64x64.flatten()
    
    # Kích hoạt truyền dữ liệu
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)
    
    # Đợi phần cứng chạy xong
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    
    # Lấy kết quả ra và reshape lại thành ảnh 2D
    return out_buffer.copy().reshape((128, 128))

# 4. LUỒNG CHẠY CHÍNH (ÔNG TỰ VIẾT THÊM CODE CẮT GHÉP NHÉ)
# image = cv2.imread('xray_image.jpg', cv2.IMREAD_GRAYSCALE)
# patches = ham_cat_anh_cua_ong(image, patch_size=64)
# high_res_patches = []

# for patch in patches:
#     result_patch = process_patch_on_fpga(patch)
#     high_res_patches.append(result_patch)

# final_image = ham_ghep_anh_cua_ong(high_res_patches)
# cv2.imwrite('output_super_res.jpg', final_image)

# Nhớ giải phóng bộ nhớ khi chạy xong toàn bộ
# in_buffer.freebuffer()
# out_buffer.freebuffer()

OUTPUT MONG ĐỢI TỪ ÔNG:
Hoàn thiện file script này với các hàm cắt/ghép ảnh hoàn chỉnh. Nhớ test trước phần cắt/ghép trên máy ông, còn phần đẩy DMA thì để trống đó, hôm nào ráp board tui với ông cùng chạy!