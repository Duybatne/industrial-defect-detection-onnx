import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.inference import ONNXInferencer

API_KEY = os.getenv("API_KEY", "industrial-defect-secret-key-2026")
MODEL_PATH = os.getenv("MODEL_PATH", "weights/model.onnx")
THRESHOLD_CONFIG_PATH = os.getenv("THRESHOLD_CONFIG_PATH", "configs/threshold_config.json")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the ONNX model at service boot."""
    try:
        get_inferencer()
        print(f"[✓] Industrial Defect Detection API started. Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Warning on startup: {e}")
    yield

app = FastAPI(
    title="Industrial Defect Detection API",
    description="High-throughput ONNX Runtime inference service for industrial surface defect inspection.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inferencer instance
inferencer: Optional[ONNXInferencer] = None


def get_inferencer() -> ONNXInferencer:
    global inferencer
    if inferencer is None:
        # Check fallback options: encrypted model -> quantized model -> base model
        model_to_load = MODEL_PATH
        if not os.path.exists(model_to_load):
            enc_path = "weights/model.onnx.enc"
            quant_path = "weights/model_quantized.onnx"
            base_path = "weights/model.onnx"
            if os.path.exists(quant_path):
                model_to_load = quant_path
            elif os.path.exists(base_path):
                model_to_load = base_path
            elif os.path.exists(enc_path):
                model_to_load = enc_path

        if os.path.exists(model_to_load):
            inferencer = ONNXInferencer(
                onnx_model_path=model_to_load,
                threshold_config_path=THRESHOLD_CONFIG_PATH
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference model not found at {MODEL_PATH} or fallback paths."
            )
    return inferencer


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verifies X-API-Key request header."""
    if not API_KEY:
        return True
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header."
        )
    return True





@app.get("/health", tags=["Monitoring"])
def health_check():
    """Service health probe endpoint."""
    global inferencer
    if inferencer is None:
        try:
            get_inferencer()
        except Exception:
            pass
    loaded = inferencer is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_path": getattr(inferencer, "model_path", None) if loaded else None,
        "threshold": getattr(inferencer, "threshold", 0.05) if loaded else None,
        "version": "1.0.0"
    }


@app.post("/v1/predict", tags=["Inference"])
async def predict_single_image(
    file: UploadFile = File(..., description="Image file (PNG, JPG, BMP)"),
    threshold: Optional[float] = Query(None, description="Optional custom decision threshold override"),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Classifies a single industrial surface image:
    - Normal (Pass) vs Defect (Reject)
    - Returns confidence score and uncertainty flag for human-in-the-loop review.
    """
    engine = get_inferencer()

    if not file.content_type.startswith("image/") and not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file."
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image file. File may be corrupted."
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = engine.predict(img_rgb, threshold=threshold)

    return JSONResponse(content={
        "prediction_id": result["prediction_id"],
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "inference_time_ms": result["inference_time_ms"],
        "uncertainty_flag": result["uncertainty_flag"],
        "is_defect": result["is_defect"],
        "threshold": result["threshold"]
    })


@app.post("/v1/predict-batch", tags=["Inference"])
async def predict_batch_images(
    files: List[UploadFile] = File(..., description="List of image files to inspect"),
    threshold: Optional[float] = Query(None, description="Optional custom decision threshold override"),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Executes high-throughput batch classification on multiple images.
    """
    engine = get_inferencer()
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in request."
        )

    decoded_images = []
    filenames = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode image file: {file.filename}"
            )
        decoded_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        filenames.append(file.filename)

    t0 = time.perf_counter()
    batch_results = engine.predict_batch(decoded_images, threshold=threshold)
    t1 = time.perf_counter()
    total_batch_time_ms = round((t1 - t0) * 1000.0, 2)

    response_items = []
    for fn, res in zip(filenames, batch_results):
        response_items.append({
            "filename": fn,
            "prediction_id": res["prediction_id"],
            "label": res["label"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "inference_time_ms": res["inference_time_ms"],
            "uncertainty_flag": res["uncertainty_flag"],
            "is_defect": res["is_defect"]
        })

    return JSONResponse(content={
        "batch_size": len(files),
        "total_inference_time_ms": total_batch_time_ms,
        "results": response_items
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deployment.app:app", host="0.0.0.0", port=8000, reload=False)
