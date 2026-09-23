import os
import sys
import time
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.inference import ONNXInferencer

st.set_page_config(
    page_title="Industrial Defect Detection - ONNX Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for industrial dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        text-align: center;
        border: 2px solid #86EFAC;
    }
    .badge-defect {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        text-align: center;
        border: 2px solid #FCA5A5;
    }
    .badge-uncertain {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        text-align: center;
        border: 2px solid #FCD34D;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inferencer(model_path: str, threshold: float):
    return ONNXInferencer(
        onnx_model_path=model_path,
        threshold_config_path="configs/threshold_config.json"
    )


# Sidebar configuration
st.sidebar.title("⚙️ Cấu Hình Suy Luận")

# Model selection
available_models = []
for p in ["weights/model.onnx", "weights/model_quantized.onnx", "weights/model.onnx.enc"]:
    if os.path.exists(p):
        available_models.append(p)

if not available_models:
    st.sidebar.error("Không tìm thấy model nào trong thư mục `weights/`.")
    selected_model = "weights/model.onnx"
else:
    selected_model = st.sidebar.selectbox("Chọn mô hình suy luận:", available_models)

# Threshold slider
threshold_val = st.sidebar.slider(
    "Ngưỡng quyết định lỗi (T*):",
    min_value=0.01,
    max_value=0.99,
    value=0.05,
    step=0.01,
    help="Ngưỡng tối ưu xác định từ Giai đoạn 2 để đạt Defect Recall >= 98%."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Chọn Ảnh Mẫu Từ Test Set")

# Sample loader
test_dir = Path("data/processed/test")
sample_choice = "Tải ảnh từ máy tính (Upload)"
sample_dict = {}

if test_dir.exists():
    good_samples = sorted(list((test_dir / "good").glob("*.png")))[:6]
    defect_samples = sorted(list((test_dir / "defect").glob("*.png")))[:6]

    for p in good_samples:
        sample_dict[f"🟢 [Good] {p.name}"] = p
    for p in defect_samples:
        sample_dict[f"🔴 [Defect] {p.name}"] = p

    options = ["Tải ảnh từ máy tính (Upload)"] + list(sample_dict.keys())
    sample_choice = st.sidebar.selectbox("Chọn ảnh kiểm thử nhanh:", options)

# Main UI Header
st.markdown('<div class="main-title">🔍 Hệ Thống Kiểm Định Ngoại Quan Công Nghiệp (Defect Detection)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ONNX Runtime Inference Engine • Tiêu chuẩn an toàn công nghiệp • Phản hồi siêu tốc dưới 40ms trên CPU</div>', unsafe_allow_html=True)

# Image input handling
image_to_inspect = None

col_img, col_res = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("1. Ảnh Đầu Vào")
    if sample_choice == "Tải ảnh từ máy tính (Upload)":
        uploaded_file = st.file_uploader("Kéo thả ảnh sản phẩm cần kiểm tra:", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            image_to_inspect = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_inspect, caption=f"Ảnh tải lên: {uploaded_file.name}", use_column_width=True)
    else:
        chosen_path = sample_dict[sample_choice]
        image_to_inspect = Image.open(chosen_path).convert("RGB")
        st.image(image_to_inspect, caption=f"Mẫu kiểm tra: {sample_choice}", use_column_width=True)

with col_res:
    st.subheader("2. Kết Quả Kiểm Định Trực Quan")
    if image_to_inspect is None:
        st.info("👈 Vui lòng chọn ảnh mẫu bên trái hoặc tải ảnh lên để bắt đầu kiểm tra.")
    else:
        # Run inference
        try:
            inferencer = load_inferencer(selected_model, threshold_val)
            np_img = np.array(image_to_inspect)

            t0 = time.perf_counter()
            result = inferencer.predict(np_img, threshold=threshold_val)
            t1 = time.perf_counter()

            latency_ms = result["inference_time_ms"]
            fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0
            is_defect = result["is_defect"]
            confidence = result["confidence"]
            defect_score = result["defect_score"]
            uncertain = result["uncertainty_flag"]

            # Visual Badge
            if uncertain:
                st.markdown(
                    f'<div class="badge-uncertain">⚠️ CẢNH BÁO: BẤT ĐỊNH ({confidence*100:.1f}%)<br><span style="font-size: 0.9rem; font-weight: normal;">Cần nhân viên kiểm định trực tiếp</span></div>',
                    unsafe_allow_html=True
                )
            elif is_defect:
                st.markdown(
                    f'<div class="badge-defect">❌ PHÁT HIỆN LỖI (DEFECT)<br><span style="font-size: 0.9rem; font-weight: normal;">Độ tin cậy lỗi: {defect_score*100:.2f}% (Vượt ngưỡng {threshold_val})</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="badge-pass">✅ ĐẠT CHUẨN (NORMAL / PASS)<br><span style="font-size: 0.9rem; font-weight: normal;">Độ tin cậy: {confidence*100:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Thời gian suy luận", f"{latency_ms:.2f} ms")
            m2.metric("Tốc độ", f"{fps} FPS")
            m3.metric("Xác suất Lỗi", f"{defect_score*100:.1f}%")
            m4.metric("Ngưỡng (T*)", f"{threshold_val}")

            st.markdown("---")
            st.markdown("#### Phân Phối Xác Suất (Softmax Probabilities)")
            prob_normal = result["probabilities"]["Normal"]
            prob_defect = result["probabilities"]["Defect"]

            st.write(f"**Bình thường (Normal):** {prob_normal*100:.2f}%")
            st.progress(float(prob_normal))

            st.write(f"**Khuyết tật (Defect):** {prob_defect*100:.2f}%")
            st.progress(float(prob_defect))

        except Exception as e:
            st.error(f"Lỗi trong quá trình suy luận: {e}")
