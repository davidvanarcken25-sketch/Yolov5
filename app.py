import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------
st.set_page_config(page_title="Detección de Objetos con YOLOv5", page_icon="🔍", layout="wide")

st.title("🔍 Detección de Objetos en Imágenes")
st.write("Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes que subas o captures con tu cámara.")

# -----------------------------
# CARGA DEL MODELO YOLOv5
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# -----------------------------
# SUBIR O CAPTURAR IMAGEN
# -----------------------------
opcion = st.radio("Selecciona una fuente de imagen:", ["📁 Subir imagen", "📷 Capturar con cámara"])

if opcion == "📁 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif opcion == "📷 Capturar con cámara":
    captured_image = st.camera_input("Toma una foto")
    if captured_image:
        image = Image.open(captured_image)
else:
    image = None

# -----------------------------
# PROCESAMIENTO DE LA IMAGEN
# -----------------------------
if model and 'image' in locals() and image:
    st.image(image, caption="Imagen original", use_container_width=True)
    st.write("Detectando objetos...")

    # Convertir imagen a formato compatible
    img_array = np.array(image)
    results = model(img_array)

    # Mostrar resultados
    st.image(np.squeeze(results.render()), caption="Resultado de detección", use_container_width=True)

    # Mostrar etiquetas y confianza
    st.subheader("Resultados detectados:")
    for *box, conf, cls in results.xyxy[0]:
        st.write(f"🟢 {model.names[int(cls)]} — Confianza: {conf:.2f}")

elif not model:
    st.warning("⚠️ No se pudo cargar el modelo. Verifica tu conexión a internet.")
