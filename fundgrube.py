import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SEITEN KONFIGURATION ---
st.set_page_config(page_title="KI Fundgrube", page_icon="🔍")

def load_model():
    # Pfad zu deinem exportierten Teachable Machine Modell
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    return model

def predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisierung (wie in Teachable Machine Standard)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array

    prediction = model.predict(data)
    return prediction

# --- UI DESIGN ---
st.title("🔍 KI-Fundgrube")
st.write("Lade ein Bild hoch und die KI sagt dir, was es ist!")

# Modell laden
model = load_model()

# Labels laden
try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = None

# Upload Bereich
file = st.file_uploader("Bild auswählen...", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Bitte lade ein Bild hoch.")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Vorhersage Button
    if st.button("Analysieren"):
        with st.spinner('KI denkt nach...'):
            prediction = predict(image, model)
            
            # Ergebnis anzeigen
            index = np.argmax(prediction)
            class_name = class_names[index] if class_names else f"Klasse {index}"
            confidence_score = prediction[0][index]

            st.success(f"Gefunden: **{class_name[2:]}**")
            st.metric("Sicherheit", f"{round(confidence_score * 100, 2)}%")