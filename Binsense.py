import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Configuración de la página de Streamlit
st.set_page_config(page_title="Binsense - Clasificador de Residuos", layout="centered")

st.title("♻️ Binsense: Detección de Residuos")
st.write("Carga una foto o usa la cámara para saber si el residuo es Reciclable o No Reciclable.")

# Cargar modelo y etiquetas de forma segura en la web
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    return model

@st.cache_resource
def load_my_labels():
    with open("etiquetas.txt", "r") as f:
        labels = f.readlines()
    return labels

model = load_my_model()
class_names = load_my_labels()

# Opción para que el usuario suba una imagen o use la cámara de su celular/PC
img_file = st.camera_input("Toma una foto al residuo")

if img_file is not None:
    # 1. Preparar la imagen
    image = Image.open(img_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Convertir imagen a array para el modelo
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 3. Hacer la predicción
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()[2:]
    confidence_score = prediction[0][index]

    # 4. Mostrar resultados con colores bonitos en la web
    st.subheader(f"Resultado: {class_name}")
    st.progress(float(confidence_score))
    st.write(f"Nivel de confianza: {confidence_score * 100:.2f}%")

    if "Reciclable" in class_name and "No" not in class_name:
        st.success("✅ ¡Este objeto es RECICLABLE! Recuerda limpiarlo antes de depositarlo.")
    else:
        st.error("❌ Este objeto NO es reciclable (Residuo Común).")

