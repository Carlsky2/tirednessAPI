from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io, time, json



import tensorflow as tf

# 4) Forzar política float32 (sin mixed precision)

model = tf.keras.models.load_model("modelo_mobilenetv2.keras")
print("Modelo cargado correctamente.")

# Si entrenaste con mixed precision, asegurate de convertir entradas a float32
policy = tf.keras.mixed_precision.global_policy()
print(f"Política activa: {policy.name}")



app = FastAPI(title="Keras Inference API (.keras)")

# 1) Carga el modelo .keras
MODEL_PATH = "modelo_mobilenetv2.keras"
model = tf.keras.models.load_model(MODEL_PATH)  # soporta Functional/Sequential/subclassed si guardaste como .keras

# 2) Intenta deducir tamaño de entrada
#    Soportamos entradas [B,H,W,C] float32 típicas
input_shape = model.inputs[0].shape  # TensorShape(None, H, W, C)
H = int(input_shape[1])
W = int(input_shape[2])
C = int(input_shape[3]) if len(input_shape) >= 4 else 3

# 3) Carga labels (opcional). Si no hay, generamos nombres genéricos
try:
    with open("labels.txt", "r", encoding="utf-8") as f:
        LABELS = [ln.strip() for ln in f if ln.strip()]
except FileNotFoundError:
    LABELS = None

def preprocess_image(img: Image.Image, norm: str = "mobilenet"):
    """
    norm:
      - "mobilenet": (x-127.5)/127.5  -> [-1,1]
      - "0_1": x/255.0                -> [0,1]
      - "imagenet": usa tf.keras.applications.mobilenet_v2.preprocess_input
    """
    img = img.convert("RGB").resize((W, H))
    x = np.asarray(img, dtype=np.float32)

    if norm == "mobilenet":
        x = (x - 127.5) / 127.5
    elif norm == "0_1":
        x = x / 255.0
    elif norm == "imagenet":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        # sin normalización extra
        pass

    x = np.expand_dims(x, axis=0)  # [1,H,W,3]
    return x

def postprocess_logits(logits: np.ndarray, topk: int = 5):
    """
    Acepta:
      - vector de logits / probabilidades [1, num_classes]
      - o predicción ya softmax
    """
    probs = logits.squeeze()
    # Si tu modelo NO tiene softmax en la última capa, puedes aplicarlo:
    if probs.ndim == 1 and not np.all((probs >= 0) & (probs <= 1)) or abs(np.sum(probs) - 1.0) > 1e-3:
        # Heurística simple: aplicar softmax
        e = np.exp(probs - np.max(probs))
        probs = e / (e.sum() + 1e-12)

    idxs = np.argsort(probs)[::-1][:topk]
    results = []
    for i in idxs:
        label = (LABELS[i] if LABELS and i < len(LABELS) else f"class_{i}")
        results.append({"index": int(i), "label": label, "score": float(probs[i])})
    return results

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    meta = {
        "input_shape": [int(d) if d is not None else None for d in model.inputs[0].shape],
        "output_shape": [int(d) if d is not None else None for d in model.outputs[0].shape],
        "labels": len(LABELS) if LABELS else 0,
        "framework": "tensorflow-keras",
        "model_path": MODEL_PATH
    }
    return JSONResponse(meta)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=50),
    norm: str = Query("mobilenet", description="mobilenet | 0_1 | imagenet | none")
):
    """
    Sube una imagen y obtiene Top-K.
    norm:
      - mobilenet: (x-127.5)/127.5
      - 0_1: x/255
      - imagenet: preprocess_input de MobileNetV2
      - none: sin normalización
    """
    t0 = time.time()
    content = await file.read()
    img = Image.open(io.BytesIO(content))

    x = preprocess_image(img, norm=norm.lower())
    # 4) Inferencia
    preds = model.predict(x)  # [1, C] normalmente
    results = postprocess_logits(np.array(preds), topk=topk)

    return JSONResponse({
        "results": results,
        "latency_ms": int((time.time() - t0) * 1000),
        "norm": norm.lower()
    })
