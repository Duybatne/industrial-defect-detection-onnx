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
from contextlib import asynccontextmanager

# Resolve base directory relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference import ONNXInferencer

API_KEY = os.getenv("API_KEY", "industrial-defect-secret-key-2026")
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "weights" / "model.onnx"),
    )
)
THRESHOLD_CONFIG_PATH = Path(
    os.getenv(
        "THRESHOLD_CONFIG_PATH",
        str(BASE_DIR / "configs" / "threshold_config.json"),
    )
)

# Global inferencer and error tracking
inferencer: Optional[ONNXInferencer] = None
model_load_error: Optional[Exception] = None


def get_inferencer() -> Optional[ONNXInferencer]:
    """Retrieves or initializes the global ONNX inferencer instance."""
    global inferencer, model_load_error
    if inferencer is None:
        candidate_paths = [
            MODEL_PATH,
            BASE_DIR / "weights" / "model.onnx",
            BASE_DIR / "weights" / "model_quantized.onnx",
            BASE_DIR / "weights" / "model.onnx.enc",
            BASE_DIR / "models" / "industrial_defect_model.onnx",
        ]
        model_to_load = None
        for p in candidate_paths:
            if Path(p).is_file():
                model_to_load = str(p)
                break

        if model_to_load is not None:
            try:
                thresh_path = str(THRESHOLD_CONFIG_PATH) if THRESHOLD_CONFIG_PATH.is_file() else None
                inferencer = ONNXInferencer(
                    onnx_model_path=model_to_load,
                    threshold_config_path=thresh_path
                )
                model_load_error = None
            except Exception as e:
                model_load_error = e
                inferencer = None
                print(f"[!] Unable to initialize ONNX inferencer: {e}")
        else:
            model_load_error = FileNotFoundError(f"Model not found at {MODEL_PATH} or fallback candidate paths.")
            inferencer = None
    return inferencer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the ONNX model at service boot."""
    try:
        engine = get_inferencer()
        if engine:
            print(f"[✓] Industrial Defect Detection API started. Model loaded: {engine.model_path}")
        else:
            print(f"[!] Service started without loaded model: {model_load_error}")
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


def validate_image(upload_file: UploadFile, contents: bytes) -> np.ndarray:
    """
    Validates uploaded file MIME type and decodes image bytes.
    Raises HTTP 400 Bad Request if file is invalid or corrupted.
    """
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}
    is_valid_type = (
        (upload_file.content_type and upload_file.content_type.lower() in allowed_types) or
        (upload_file.filename and upload_file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    )

    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{upload_file.content_type}'. Please upload an image file."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image: uploaded file is empty."
        )

    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image: file may be corrupted."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Service health probe endpoint."""
    engine = get_inferencer()
    loaded = engine is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_path": getattr(engine, "model_path", None) if loaded else None,
        "threshold": getattr(engine, "threshold", 0.05) if loaded else None,
        "version": "1.0.0",
        "error": str(model_load_error) if not loaded else None
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
    contents = await file.read()

    # Validate the request before checking model availability.
    img_rgb = validate_image(file, contents)

    engine = get_inferencer()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {model_load_error}"
        )

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
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in request."
        )

    decoded_images = []
    filenames = []
    for file in files:
        contents = await file.read()
        # Validate the request before checking model availability.
        img_rgb = validate_image(file, contents)
        decoded_images.append(img_rgb)
        filenames.append(file.filename)

    engine = get_inferencer()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {model_load_error}"
        )

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
