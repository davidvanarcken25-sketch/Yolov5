import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Configuración general
st.set_page_config(page_title="Detección de Objetos YOLOv5", page_icon="🔍", layout="wide")

st.title("🔍 Detección de Objetos en Imágenes")
st.write("""
Esta aplicación usa **YOLOv5** para detectar objetos en imágenes.  
Sube una imagen o usa tu cámara para probarlo.
""")

# Cargar modelo (desde torch.hub)
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

model = load_model()

# Subir o capturar imagen
opcion = st.radio("Selecciona la fuente de la imagen:", ["📁 Subir imagen", "📷 Capturar con cámara"])

if opcion == "📁 Subir imagen":
    archivo = st.file_uploader("Sube una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if archivo:
        imagen = Image.open(archivo)
elif opcion == "📷 Capturar con cámara":
    captura = st.camera_input("Toma una foto")
    if captura:
        imagen = Image.open(captura)
else:
    imagen = None

# Procesar la imagen
if model and "imagen" in locals() and imagen:
    st.image(imagen, caption="Imagen original", use_container_width=True)
    st.write("Analizando imagen... 🔄")

    img_array = np.array(imagen)
    resultados = model(img_array)

    # Mostrar resultado
    st.image(np.squeeze(resultados.render()), caption="Resultado de detección", use_container_width=True)

    # Mostrar etiquetas
    st.subheader("Resultados detectados:")
    for *box, conf, cls in resultados.xyxy[0]:
        st.write(f"🟢 {model.names[int(cls)]} — Confianza: {conf:.2f}")

elif not model:
    st.warning("⚠️ No se pudo cargar el modelo. Verifica tu conexión a Internet.")
