# Bảng Đo Kiểm Hiệu Năng Suy Luận (Benchmark Table)

- **Môi trường đo kiểm:** CPU (Intel/AMD x86_64, 4 intra-op threads)
- **Cấu hình:** Batch Size = 1 | Warmup = 50 iterations | Đo kiểm = 200 iterations
- **Kích thước ảnh đầu vào:** (1, 3, 224, 224)

| Mô hình / Định dạng | Dung lượng (MB) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (FPS) | Tăng tốc (Speedup) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32** | 46.34 MB | 92.57 ms | 145.04 ms | 225.8 ms | 10.4 FPS | **1.0x** |
| **ONNX FP32** | 15.28 MB | 39.4 ms | 141.42 ms | 215.64 ms | 17.5 FPS | **2.35x** |
| **ONNX INT8** | 15.29 MB | 34.45 ms | 86.79 ms | 127.25 ms | 23.8 FPS | **2.69x** |
