import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil
import torch
import yaml
import onnxruntime as ort

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.network import build_model


def get_process_memory_mb() -> float:
    """Returns the current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_pytorch(
    weights_path: str = "weights/best_model.pth",
    config_path: str = "configs/train_config.yaml",
    img_size: tuple = (224, 224),
    batch_size: int = 1,
    warmup: int = 50,
    iterations: int = 200
) -> Dict[str, Any]:
    """Benchmarks PyTorch FP32 CPU inference."""
    mem_before = get_process_memory_mb()

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {"backbone": "efficientnet_b0", "num_classes": 2, "pretrained": False, "dropout": 0.3}}

    model = build_model(config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1], dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mem_after = get_process_memory_mb()
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    return {
        "framework": "PyTorch FP32",
        "file_size_mb": round(file_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(float(1000.0 * batch_size / np.mean(latencies)), 1),
        "ram_mb": round(mem_after - mem_before, 2)
    }


def benchmark_onnx(
    model_path: str,
    name: str = "ONNX FP32",
    img_size: tuple = (224, 224),
    batch_size: int = 1,
    warmup: int = 50,
    iterations: int = 200
) -> Dict[str, Any]:
    """Benchmarks ONNX Runtime CPU inference."""
    mem_before = get_process_memory_mb()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_input = np.random.randn(batch_size, 3, img_size[0], img_size[1]).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mem_after = get_process_memory_mb()
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return {
        "framework": name,
        "file_size_mb": round(file_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(float(1000.0 * batch_size / np.mean(latencies)), 1),
        "ram_mb": round(mem_after - mem_before, 2)
    }


def run_benchmarks(
    torch_weights: str = "weights/best_model.pth",
    onnx_model: str = "weights/model.onnx",
    quantized_model: str = "weights/model_quantized.onnx",
    config_path: str = "configs/train_config.yaml",
    warmup: int = 50,
    iterations: int = 200,
    report_json_path: str = "reports/benchmark_summary.json",
    report_md_path: str = "reports/benchmark_table.md"
) -> List[Dict[str, Any]]:
    """Runs comprehensive comparative latency & resource benchmarking."""
    print("=" * 72)
    print("TIẾN HÀNH BENCHMARK HIỆU NĂNG SUY LUẬN (LATENCY & RESOURCE BENCHMARK)")
    print(f"  - Thiết bị: CPU (x86_64, 4 threads)")
    print(f"  - Batch Size: 1 | Warmup: {warmup} runs | Đo kiểm: {iterations} runs")
    print("=" * 72)

    results = []

    # 1. PyTorch FP32
    if os.path.exists(torch_weights):
        print("[1/3] Benchmarking PyTorch FP32...")
        res_torch = benchmark_pytorch(
            weights_path=torch_weights,
            config_path=config_path,
            warmup=warmup,
            iterations=iterations
        )
        results.append(res_torch)
        print(f"    - P50: {res_torch['p50_latency_ms']} ms | FPS: {res_torch['fps']} | Size: {res_torch['file_size_mb']} MB")

    # 2. ONNX FP32
    if os.path.exists(onnx_model):
        print("[2/3] Benchmarking ONNX Runtime FP32...")
        res_onnx = benchmark_onnx(
            model_path=onnx_model,
            name="ONNX FP32",
            warmup=warmup,
            iterations=iterations
        )
        results.append(res_onnx)
        print(f"    - P50: {res_onnx['p50_latency_ms']} ms | FPS: {res_onnx['fps']} | Size: {res_onnx['file_size_mb']} MB")

    # 3. ONNX INT8
    if os.path.exists(quantized_model):
        print("[3/3] Benchmarking ONNX Runtime INT8...")
        res_quant = benchmark_onnx(
            model_path=quantized_model,
            name="ONNX INT8",
            warmup=warmup,
            iterations=iterations
        )
        results.append(res_quant)
        print(f"    - P50: {res_quant['p50_latency_ms']} ms | FPS: {res_quant['fps']} | Size: {res_quant['file_size_mb']} MB")

    # Calculate speedup relative to PyTorch
    base_latency = results[0]["p50_latency_ms"] if results else 1.0
    for r in results:
        r["speedup"] = round(base_latency / r["p50_latency_ms"], 2) if r["p50_latency_ms"] > 0 else 1.0

    # Build Markdown Table
    md_content = f"""# Bảng Đo Kiểm Hiệu Năng Suy Luận (Benchmark Table)

- **Môi trường đo kiểm:** CPU (Intel/AMD x86_64, 4 intra-op threads)
- **Cấu hình:** Batch Size = 1 | Warmup = {warmup} iterations | Đo kiểm = {iterations} iterations
- **Kích thước ảnh đầu vào:** (1, 3, 224, 224)

| Mô hình / Định dạng | Dung lượng (MB) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (FPS) | Tăng tốc (Speedup) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        md_content += f"| **{r['framework']}** | {r['file_size_mb']} MB | {r['p50_latency_ms']} ms | {r['p95_latency_ms']} ms | {r['p99_latency_ms']} ms | {r['fps']} FPS | **{r['speedup']}x** |\n"

    # Save reports
    os.makedirs(os.path.dirname(os.path.abspath(report_json_path)), exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_results": results}, f, indent=2)
    print(f"\n[✓] Đã lưu tóm tắt benchmark tại: {report_json_path}")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Đã lưu bảng Markdown tại: {report_md_path}\n")

    # Print to console
    print(md_content)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX models")
    parser.add_argument("--torch-weights", default="weights/best_model.pth", help="Path to PyTorch weights")
    parser.add_argument("--onnx-model", default="weights/model.onnx", help="Path to ONNX model")
    parser.add_argument("--quantized-model", default="weights/model_quantized.onnx", help="Path to INT8 model")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Number of test iterations")
    args = parser.parse_args()

    run_benchmarks(
        torch_weights=args.torch_weights,
        onnx_model=args.onnx_model,
        quantized_model=args.quantized_model,
        config_path=args.config,
        warmup=args.warmup,
        iterations=args.iterations
    )
