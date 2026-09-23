import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.pipeline.inference import ONNXInferencer

app = FastAPI(title="Industrial Defect Detection API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "weights/model.onnx")
inferencer = None


@app.on_event("startup")
def load_model():
    global inferencer
    if os.path.exists(MODEL_PATH):
        inferencer = ONNXInferencer(MODEL_PATH)
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. API will return error on predict.")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": inferencer is not None}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or weights/model.onnx missing.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = inferencer.predict(img_rgb)
    return JSONResponse(content=result)
