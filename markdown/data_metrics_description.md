# Thống Kê Các Bộ Dữ Liệu Kiểm Thử (Evaluation Datasets Metrics)

Tài liệu này cung cấp thống kê chi tiết về **3 bộ dữ liệu X-quang ngực (Chest X-Ray)** được hợp nhất vào thư mục `X-Ray images/` phục vụ cho việc đánh giá và kiểm nghiệm (Inference & Evaluation) mô hình Siêu độ phân giải ảnh y tế Swift-SRGAN trên cả mô phỏng phần mềm (CPU/GPU) và tăng tốc phần cứng (FPGA).

---

## 1. Bảng Tổng Hợp Thống Kê (Summary Table)

| Tên Bộ Dữ Liệu | Số Lượng Ảnh | Tổng Dung Lượng | Định Dạng Ảnh | Độ Phân Giải Gốc | Đặc Điểm Lâm Sàng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NIH ChestX-ray14** | 5.157 ảnh | 1,90 GB | PNG (Grayscale) | $1024 \times 1024$ | 14 loại bệnh lý lồng ngực phổ biến ở người lớn |
| **COVID-19 Radiography** | 9.628 ảnh | 325,57 MB | PNG (Grayscale) | $299 \times 299$ | Phổi bình thường, nhiễm Covid-19, Viêm phổi virus |
| **Guangzhou Pediatric Pneumonia** | 5.856 ảnh | 1,15 GB | JPEG (Grayscale/RGB) | Dao động từ $1000 \times 800$ đến $2000 \times 1500$ | Phổi bình thường và viêm phổi ở trẻ em (1-5 tuổi) |

---

## 2. Chi Tiết Từng Bộ Dữ Liệu & Tài Liệu Trích Dẫn (Citation)

### 2.1. NIH ChestX-ray14 (National Institutes of Health)
*   **Đường dẫn gốc:** [Kaggle - NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) | [Official Box Folder](https://nihcc.app.box.com/v/ChestXray-NIHCC)
*   **Tổng quan:** Bộ dữ liệu y tế quy mô lớn được thu thập từ Trung tâm Y tế Lâm sàng NIH (Mỹ). Sử dụng phần kiểm thử độc lập gồm 5.157 ảnh chưa từng xuất hiện trong tập huấn luyện (90.000 ảnh) để đảm bảo không bị rò rỉ dữ liệu (no data leakage).
*   **Đặc điểm hình ảnh:** Ảnh chụp X-quang ngực thẳng người lớn, có độ tương phản chuẩn y khoa cao, thang màu xám chuẩn nét giúp đánh giá tối ưu chất lượng tái tạo mô xương và mô phế quản phổi.
*   **Tài liệu trích dẫn khoa học (IEEE Reference Format):**
    ```text
    X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri and R. M. Summers, "ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3462-3471, 2017.
    ```
*   **Mẫu BibTeX:**
    ```bibtex
    @inproceedings{wang2017chestxray,
      title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
      author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={3462--3471},
      year={2017}
    }
    ```

---

### 2.2. COVID-19 Radiography Database (Qatar University & University of Dhaka)
*   **Đường dẫn gốc:** [Kaggle - COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
*   **Tổng quan:** Bộ dữ liệu đoạt giải thưởng của nhóm nghiên cứu Qatar và Bangladesh. Được tạo ra nhằm mục đích phát hiện nhanh các tổn thương do COVID-19 và các loại viêm phổi khác thông qua ảnh X-quang.
*   **Đặc điểm hình ảnh:** Các ảnh đã được tiền xử lý chuẩn hóa kích thước về $299 \times 299$ pixel, định dạng PNG, ảnh sắc nét giúp kiểm chứng hiệu năng xử lý của mô hình trên các ảnh có độ phân giải trung bình.
*   **Tài liệu trích dẫn khoa học (IEEE Reference Format):**
    ```text
    M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al-Emadi, and M. B. I. Reaz, "Can AI help in screening viral and COVID-19 pneumonia?" IEEE Access, vol. 8, pp. 132665–132676, 2020.
    ```
*   **Mẫu BibTeX:**
    ```bibtex
    @article{chowdhury2020can,
      title={Can AI help in screening viral and COVID-19 pneumonia?},
      author={Chowdhury, Muhammad EH and Rahman, Tawsifur and Khandakar, Amith and Mazhar, Ruqayya and Kadir, Muhammad Abdul and Mahbub, Zaid Bin and Islam, Kazi Rogers and Khan, Muhammad Salman and Iqbal, Atif and Al-Emadi, Nasser and others},
      journal={IEEE Access},
      volume={8},
      pages={132665--132676},
      year={2020},
      publisher={IEEE}
    }
    ```

---

### 2.3. Guangzhou Pediatric Pneumonia Dataset (Guangzhou Women and Children's Medical Center)
*   **Đường dẫn gốc:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
*   **Tổng quan:** Bộ dữ liệu X-quang phổi nhi khoa của Bệnh viện Phụ sản và Nhi đồng Quảng Châu. Được kiểm duyệt lâm sàng bởi các bác sĩ chẩn đoán hình ảnh giàu kinh nghiệm.
*   **Đặc điểm hình ảnh:** Độ phân giải ảnh không đồng đều (nhiều ảnh kích thước rất lớn), chứa một số vùng nhiễu thực tế lâm sàng. Cực kỳ hữu dụng để đánh giá độ bền (Robustness) của thuật toán chia mảnh (Tiling) và ghép nối (Stitching) của file Host khi xử lý ảnh kích thước biến động lớn.
*   **Tài liệu trích dẫn khoa học (IEEE Reference Format):**
    ```text
    D. S. Kermany, M. Goldbaum, W. Cai, et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, pp. 1122-1131, 2018.
    ```
*   **Mẫu BibTeX:**
    ```bibtex
    @article{kermany2018identifying,
      title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
      author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and McKeown, Alex and Yang, Ge and Teibert, Xiaokang and Yan, Fangbing and others},
      journal={Cell},
      volume={172},
      number={5},
      pages={1122--1131},
      year={2018},
      publisher={Elsevier}
    }
    ```
