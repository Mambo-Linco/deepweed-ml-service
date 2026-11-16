"""
DeepWeed AI model service.

Implements steps 4-7 of the system flow:
  4 preprocess(image)  ->  5 predict(preprocessed)  ->  6 forwardPass()  ->  7 top3Species(conf)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger("deepweed-ml")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_FILE = os.getenv("MODEL_FILE", "weed_classifier.h5")
IMG_SIZE = (224, 224)  # from the model's input shape (None, 224, 224, 3)

# The saved model is a MobileNetV2 transfer-learning head with NO rescaling layer inside,
# so inputs must be scaled the same way as during training:
#   "rescale"   -> pixels / 255            (ImageDataGenerator(rescale=1/255), the most common setup)
#   "mobilenet" -> pixels / 127.5 - 1      (tf.keras.applications.mobilenet_v2.preprocess_input)
PREPROCESS = os.getenv("PREPROCESS", "rescale").lower()

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    import tensorflow as tf  # imported here so --help / import checks stay fast

    names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    indices = json.loads((MODEL_DIR / "class_indices.json").read_text(encoding="utf-8"))
    # class_indices.json is the authoritative index -> name mapping from training
    by_index = {v: k for k, v in indices.items()}
    ordered = [by_index[i] for i in range(len(by_index))]
    if ordered != names:
        log.warning("class_names.json order differs from class_indices.json; using class_indices.json")

    log.info("Loading model %s ...", MODEL_DIR / MODEL_FILE)
    model = tf.keras.models.load_model(MODEL_DIR / MODEL_FILE, compile=False)
    n_out = model.output_shape[-1]
    if n_out != len(ordered):
        raise RuntimeError(f"Model has {n_out} outputs but {len(ordered)} class names were found")

    # warm-up so the first real request is fast
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)
    state.update(model=model, classes=ordered)
    log.info("Model ready: %d classes, preprocess=%s", n_out, PREPROCESS)
    yield
    state.clear()


app = FastAPI(title="DeepWeed ML service", version="1.0.0", lifespan=lifespan)


def preprocess(raw: bytes) -> np.ndarray:
    """Step 4: decode, fix orientation, RGB, resize to 224x224, scale."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unreadable image") from exc
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    x = np.asarray(img, dtype="float32")
    x = x / 127.5 - 1.0 if PREPROCESS == "mobilenet" else x / 255.0
    return np.expand_dims(x, 0)


@app.get("/health")
def health():
    ready = "model" in state
    return {
        "status": "ok" if ready else "loading",
        "model": MODEL_FILE,
        "classes": len(state.get("classes", [])),
        "preprocess": PREPROCESS,
    }


@app.post("/predict")
async def predict(request: Request):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is still loading")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body")

    t0 = time.perf_counter()
    x = preprocess(raw)                                   # 4
    probs = state["model"].predict(x, verbose=0)[0]       # 5/6 forward pass
    top = np.argsort(probs)[::-1][:3]                     # 7 top-3
    classes = state["classes"]
    return {
        "model": MODEL_FILE,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        "top3": [
            {
                "class_index": int(i),
                "class_name": classes[int(i)],
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in top
        ],
    }
