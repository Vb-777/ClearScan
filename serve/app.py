"""ClearScan inference service (research prototype - not a medical device).

    uvicorn serve.app:app --port 8000
    curl -F image=@lesion.jpg localhost:8000/predict

Runs the INT8 ONNX model with onnxruntime. Probabilities are temperature-calibrated, and the
melanoma referral flag uses the threshold chosen on validation data (models/serving.json).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError

ROOT = Path(os.environ.get("CLEARSCAN_HOME", Path(__file__).resolve().parent.parent))
CFG = json.load(open(ROOT / "models" / "serving.json"))
MODEL_FILE = ROOT / "models" / CFG["model_file"]
so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("CLEARSCAN_THREADS", "1"))
SESSION = ort.InferenceSession(str(MODEL_FILE), so, providers=["CPUExecutionProvider"])
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="ClearScan", version=CFG["version"],
              description="Dermatoscopic lesion classifier (7 classes). Research prototype, not for clinical use.")


def preprocess(raw: bytes) -> np.ndarray:
    """Any RGB image -> 1 x 3 x 28 x 28 float32 in 0-255, resized like DermaMNIST (whole image, bicubic)."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="file is not a readable image")
    if min(im.size) < 28:
        raise HTTPException(status_code=422, detail="image smaller than 28 px")
    im = im.convert("RGB").resize((28, 28), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]


def classify(x: np.ndarray) -> dict:
    logits = SESSION.run(None, {"image": x})[0][0] / CFG["temperature"]
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    classes = CFG["classes"]
    top = int(p.argmax())
    mel = float(p[classes.index("mel")])
    return {
        "prediction": classes[top], "prediction_name": CFG["class_names"][classes[top]],
        "confidence": float(p[top]),
        "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
        "melanoma_probability": round(mel, 4),
        "refer_for_melanoma_review": mel >= CFG["melanoma_threshold"],
        "low_confidence": float(p[top]) < CFG["low_confidence_below"],
        "model_version": CFG["version"],
        "notice": "Research prototype trained on 28x28 HAM10000 images. Not for diagnosis.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG["model_file"], "version": CFG["version"]}


@app.get("/model-card", response_class=PlainTextResponse)
def model_card():
    return (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    raw = await image.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image larger than 10 MB")
    return classify(preprocess(raw))
