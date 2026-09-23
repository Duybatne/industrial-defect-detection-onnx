# Industrial Defect Detection with ONNX Runtime

Phát hiện và phân loại lỗi bề mặt sản phẩm công nghiệp (kim loại, bo mạch PCB, dệt may) tối ưu hóa suy luận tốc độ cao với ONNX Runtime.

## 1. Tính Năng Chính
- Pipeline phân loại ảnh End-to-End với PyTorch và Timm.
- Data Augmentation tối ưu cho phát hiện lỗi với Albumentations.
- Export mô hình sang ONNX Runtime phục vụ suy luận CPU/GPU độ trễ thấp.
- REST API đóng gói sẵn bằng FastAPI & Docker container.
- Unit testing và CI/CD GitHub Actions sẵn sàng.

## 2. Cấu Trúc Dự Án
```text
industrial-defect-detection-onnx/
├── .github/workflows/       # CI/CD test tự động
├── configs/                 # Cấu hình hyperparameters
├── data/                    # Thư mục dữ liệu raw/processed
├── deployment/              # FastAPI app, ONNX export, Dockerfile
├── notebooks/               # EDA & benchmarking
├── src/                     # Source code (models, dataset, pipeline, utils)
├── tests/                   # Unit test
├── weights/                 # Checkpoints & ONNX weights
├── requirements.txt
└── README.md
```

## 3. Cài Đặt & Chạy Thử
```bash
pip install -r requirements.txt
```

### Export ONNX
```bash
python deployment/export_onnx.py --config configs/train_config.yaml
```

### Chạy API Service
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```
