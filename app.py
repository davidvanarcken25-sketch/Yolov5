import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- Título y descripción ---
st.set_page_config(page_title="Detección de Objetos", layout="wide")
st.title("🔍 Detección de Objetos en Imágenes")
st.write("Esta aplicación usa **YOLOv8** para detectar objetos. Sube una imagen o usa tu cámara para probarlo.")

# --- Carga del modelo ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")  # modelo liviano preentrenado
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

model = load_model()

# --- Fuente de la imagen ---
option = st.radio("Selecciona la fuente de la imagen:", ("Subir imagen", "Capturar con cámara"))

img = None
if option == "Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
elif option == "Capturar con cámara":
    camera_file = st.camera_input("Captura una imagen")
    if camera_file:
        img = Image.open(camera_file)

# --- Detección ---
if img is not None and model:
    st.image(img, caption="Imagen Original", use_container_width=True)

    with st.spinner("Detectando objetos..."):
        results = model.predict(np.array(img))
        annotated_img = results[0].plot()  # dibuja las cajas
        st.image(annotated_img, caption="Resultado de Detección", use_container_width=True)

    st.success("✅ Detección completada con éxito.")
else:
    st.info("Sube o captura una imagen para comenzar.")
