import cv2
import tensorflow as tf
import numpy as np

# ================== CONFIG ==================
MODEL_PATH = "modelo_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["alert", "Microslept", "yawn"]  # 🔹 Ajustá según tus carpetas

# ================== CARGAR MODELO ==================
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente.")

# Si entrenaste con mixed precision, asegurate de convertir entradas a float32
policy = tf.keras.mixed_precision.global_policy()
print(f"Política activa: {policy.name}")

# ================== CAPTURA DE CÁMARA ==================
cap = cv2.VideoCapture(0)  # 0 = cámara principal

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("🎥 Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error leyendo la cámara.")
        break

    # Preprocesamiento
    img = cv2.resize(frame, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalizar
    img = np.expand_dims(img, axis=0)     # agregar batch

    # Predicción
    preds = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    # Mostrar resultado si supera cierto umbral
    if confidence > 0.7:
        label = f"{CLASS_NAMES[pred_idx]} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if pred_idx == 2 else (0, 0, 255)
    else:
        label = "Ninguna expresión detectada"
        color = (200, 200, 200)

    # Mostrar en pantalla
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("Detección de expresiones", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== LIMPIAR ==================
cap.release()
cv2.destroyAllWindows()