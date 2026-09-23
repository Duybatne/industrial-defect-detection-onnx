# Kế Hoạch Thực Thi & Tiêu Chuẩn Kỹ Thuật: Industrial Defect Detection with ONNX

Dự án: `industrial-defect-detection-onnx`  
Mục tiêu: Xây dựng hệ thống phát hiện lỗi bề mặt sản phẩm công nghiệp (MVTec AD / PCB Defect), tối ưu hóa suy luận tốc độ cao với ONNX Runtime, đóng gói REST API và đảm bảo tiêu chuẩn an toàn công nghiệp.

---

## 1. Chi Tiết Các Giai Đoạn & Ma Trận Xử Lý Vấn Đề

### Giai Đoạn 1: Chuẩn Bị & Tiền Xử Lý Dữ Liệu

| Bước | Vấn Đề Kỹ Thuật | Giải Pháp Theo Tình Huống | Giải Pháp Tốt Nhất |
| :--- | :--- | :--- | :--- |
| **1.1 Thu thập Dataset** | Mất cân bằng dữ liệu nghiêm trọng (ảnh bình thường >> ảnh lỗi). | - Mẫu lỗi < 50 ảnh: Dùng Unsupervised Anomaly Detection (PatchCore, PaDiM).<br>- Mẫu lỗi > 200 ảnh: Phân loại nhị phân/đa lớp có giám sát. | **PatchCore** (nếu dữ liệu thực tế ít mẫu lỗi) hoặc **Focal Loss + Weighted Sampler** (nếu đủ nhãn). |
| **1.2 Pipeline Augmentation** | Mô hình học thuộc nền ảnh thay vì học đặc trưng lỗi; rò rỉ dữ liệu. | - Augmentation vật lý: CutMix, MixUp, ColorJitter, CLAHE.<br>- Phân chia tập dữ liệu: Stratified GroupKFold theo lô sản xuất (Lot/Batch ID). | **Stratified Group Split + CLAHE + Synthetic Defect Injection (CutPaste)**. |

---

### Giai Đoạn 2: Huấn Luyện & Tối Ưu Mô Hình (PyTorch)

| Bước | Vấn Đề Kỹ Thuật | Giải Pháp Theo Tình Huống | Giải Pháp Tốt Nhất |
| :--- | :--- | :--- | :--- |
| **2.1 Lựa chọn Kiến trúc** | Mô hình quá nặng gây trễ cao trên Edge; mô hình quá nhẹ thiếu độ nhạy. | - CPU Edge: MobileNetV3-Large, EfficientNet-B0.<br>- GPU Server: ResNet50, ConvNeXt-Tiny, PatchCore (ResNet50 feature extractor). | **EfficientNet-B0 (Transfer Learning) + AdamW + Cosine Annealing LR**. |
| **2.2 Hàm Mất Mát & Metric** | Bỏ sót lỗi (False Negative) gây thiệt hại lớn trong sản xuất; Accuracy bị sai lệch do mất cân bằng. | - Tối ưu Loss: Weighted CrossEntropy, Focal Loss, Asymmetric Loss.<br>- Đánh giá: PR-AUC, F1-Score, Confusion Matrix. | **Focal Loss + Tinh chỉnh Decision Threshold để Defect Recall >= 98.5%**. |

---

### Giai Đoạn 3: Tối Ưu Hóa & Đóng Gói Suy Luận (ONNX Runtime)

| Bước | Vấn Đề Kỹ Thuật | Giải Pháp Theo Tình Huống | Giải Pháp Tốt Nhất |
| :--- | :--- | :--- | :--- |
| **3.1 Export & Quantization** | Lệch kết quả giữa PyTorch và ONNX; giảm độ chính xác khi ép kiểu số nguyên (INT8). | - Opset Version: Sử dụng Opset 17 hoặc 18.<br>- CPU Target: INT8 Static Quantization (Calibration Dataset).<br>- GPU Target: FP16 TensorRT Provider. | **ONNX Opset 17 + FP16 (GPU) hoặc INT8 Static Calibration (CPU)**. Kiểm tra sai số $\Delta < 10^{-4}$. |
| **3.2 Tối Ưu Độ Trễ (Latency)** | Overhead khi copy memory giữa RAM và VRAM; I/O bound. | - Dynamic Axes nếu kích thước ảnh thay đổi.<br>- Cố định input shape (Static Shape) nếu camera cố định. | **Static Input Shape (1x3x224x224) + ONNX Runtime IOBinding**. |

---

### Giai Đoạn 4: Đóng Gói API & Trực Quan Hóa

| Bước | Vấn Đề Kỹ Thuật | Giải Pháp Theo Tình Huống | Giải Pháp Tốt Nhất |
| :--- | :--- | :--- | :--- |
| **4.1 REST API Service** | Nghẽn xử lý ảnh đồng thời (Concurrent requests); rò rỉ bộ nhớ OpenCV. | - Đồng bộ: FastAPI + ThreadPoolExecutor cho tiền xử lý CPU.<br>- Bất đồng bộ: Worker Queue (Celery/Redis) cho batch inference. | **FastAPI Async + ONNX Runtime Inference Session Pool**. |
| **4.2 Giao Diện Demo** | Cần giao diện trực quan hóa vùng lỗi và so sánh tốc độ FPS. | - Streamlit hoặc Gradio upload ảnh và render heatmap/bounding box. | **Streamlit App tích hợp so sánh PyTorch vs ONNX Latency & Confidence Score**. |

---

## 2. Bảo Mật & Tiêu Chuẩn Triển Khai Công Nghiệp

1. **Bảo Vệ Trí Tuệ Nhân Tạo (Model Weight Protection):**
   - Không lưu file `.onnx` ở dạng plaintext trên thiết bị biên (Edge).
   - Mã hóa model bằng **AES-256-CBC**, khóa bí mật nạp từ biến môi trường/Hardware Security Module (HSM).
   - Giải mã trực tiếp vào bộ nhớ RAM (`ort.InferenceSession(bytes)`) lúc khởi động ứng dụng.

2. **Bảo Mật API & Hệ Thống Mạng:**
   - Giới hạn quyền truy cập qua API Key hoặc mTLS (Mutual TLS) trong mạng nội bộ nhà máy (OT Network).
   - Chạy ứng dụng trong Docker container với user không có quyền root (`non-root user`).

3. **Giám Sát Vận Hành (Drift Detection & Reliability):**
   - **Data Drift:** Ghi nhận thống kê phân phối điểm tin cậy (Confidence Distribution). Cảnh báo khi trung bình tin cậy giảm > 15%.
   - **Human-in-the-loop:** Thiết lập vùng không chắc chắn (Uncertainty Margin: $0.40 \le P(\text{Defect}) \le 0.65$). Đưa ảnh vào hàng đợi cho kiểm định viên xác nhận thủ công.

---

## 3. Lộ Trình Thực Hiện (Checklist)

- [ ] **Sprint 1: Dữ liệu & Baseline**
  - [ ] Tải dataset MVTec AD (ví dụ category `bottle` / `pcb`).
  - [ ] Viết script chia tập train/val và EDA (`notebooks/01_eda.ipynb`).
  - [ ] Train mô hình PyTorch baseline đạt F1-Score $\ge 0.92$.
- [ ] **Sprint 2: ONNX & Optimization**
  - [ ] Export mô hình PyTorch sang ONNX format.
  - [ ] Đo kiểm Benchmark Latency (PyTorch vs ONNX vs ONNX-Quantized).
  - [ ] Viết pipeline suy luận hoàn chỉnh (`src/pipeline/inference.py`).
- [ ] **Sprint 3: Deployment & Delivery**
  - [ ] Hoàn thiện API FastAPI (`deployment/app.py`).
  - [ ] Đóng gói Dockerfile và kiểm thử CI/CD GitHub Actions.
  - [ ] Cập nhật kết quả benchmark vào `README.md`.
