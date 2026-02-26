import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client, Client
import uuid
import io
import pandas as pd
from datetime import datetime

# --- 1. SEITEN KONFIGURATION & DARKMODE ENFORCEMENT ---
st.set_page_config(page_title="KI Fundgrube Pro", page_icon="🔍", layout="wide")

# CSS für dauerhaften Darkmode & Styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #464b5d; }
        .stTextInput>div>div>input { background-color: #262730; color: white; }
        /* Verstecke den Lightmode-Umschalter optisch weitestgehend */
        [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SUPABASE SETUP ---
# Ersetze diese mit deinen echten Secrets/Environment Variablen
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "DEINE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "DEIN_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 3. KI MODELL FUNKTIONEN ---
@st.cache_resource
def load_model():
    # Keras 2.15/3.0 Kompatibilität beachten
    return tf.keras.models.load_model('keras_model.h5', compile=False)

def predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array
    return model.predict(data)

# --- 4. LOGIK FÜR FEATURES (MATCHING, FILTER, ETC.) ---

def check_for_matches(new_label, current_type):
    # Einfaches Matching (Feature #1)
    target_type = "search" if current_type == "found" else "found"
    matches = supabase.table("items").select("*").eq("label", new_label).eq("type", target_type).execute()
    return matches.data

# --- 5. UI SIDEBAR (NAVIGATION) ---
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio("Bereich wählen", 
    ["🏠 Home", "📤 Etwas melden (KI)", "📦 Fundbüro durchsuchen", "📊 Statistik & Map"])

model = load_model()
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# --- MODUS: HOME ---
if menu == "🏠 Home":
    st.title("Willkommen bei der KI-Fundgrube")
    st.write("Die intelligente Lösung für verlorene Gegenstände.")
    col1, col2 = st.columns(2)
    col1.metric("Gefundene Objekte", "124") # Beispielwerte für Feature #19/20
    col2.metric("Erfolgreiche Matches", "42")

# --- MODUS: MELDEN (FINDER/SUCHER) ---
elif menu == "📤 Etwas melden (KI)":
    st.header("Objekt erfassen")
    
    mode = st.segmented_control("Dein Status", ["Ich habe etwas gefunden", "Ich vermisse etwas"], default="Ich habe etwas gefunden")
    db_type = "found" if "gefunden" in mode else "search"
    
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Vorschau", width=300)
        
        with col2:
            st.subheader("KI-Analyse")
            if st.button("Objekt identifizieren"):
                prediction = predict(image, model)
                idx = np.argmax(prediction)
                label = class_names[idx][2:]
                conf = prediction[0][idx]
                
                st.session_state['detected_label'] = label
                st.success(f"Erkannt als: **{label}** ({round(conf*100,1)}%)")
                
                # Automatische Tags (Feature #7)
                suggested_tags = f"{label}, {datetime.now().strftime('%Y')}, Fundstück"
                st.session_state['tags'] = st.text_input("Tags bestätigen/erweitern", suggested_tags)

        if 'detected_label' in st.session_state:
            # Belohnung & Standort (Feature #11, #2)
            reward = st.number_input("Finderlohn / Belohnung (€)", min_value=0, value=0)
            location = st.text_input("Ort (PLZ oder Stadtteil)")
            
            if st.button("In Datenbank speichern"):
                # Upload zu Supabase Storage
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                file_path = f"{db_type}/{uuid.uuid4()}.jpg"
                supabase.storage.from_("images").upload(file_path, img_byte_arr.getvalue())
                img_url = supabase.storage.from_("images").get_public_url(file_path)

                # DB Eintrag
                new_item = {
                    "label": st.session_state['detected_label'],
                    "tags": [t.strip() for t in st.session_state['tags'].split(",")],
                    "image_url": img_url,
                    "type": db_type,
                    "location": location,
                    "reward": reward
                }
                supabase.table("items").insert(new_item).execute()
                
                # Matching Check (Feature #1)
                matches = check_for_matches(st.session_state['detected_label'], db_type)
                if matches:
                    st.balloons()
                    st.info(f"🎉 Treffer! Es gibt {len(matches)} passende Inserate in der Datenbank!")

# --- MODUS: DURCHSUCHEN ---
elif menu == "📦 Fundbüro durchsuchen":
    st.header("Fundstücke & Gesuche")
    
    search_query = st.text_input("🔍 Nach Tags oder Objekten suchen...")
    
    res = supabase.table("items").select("*").order("created_at", desc=True).execute()
    items = res.data
    
    if search_query:
        items = [i for i in items if search_query.lower() in str(i['tags']).lower() or search_query.lower() in i['label'].lower()]

    # Grid Ansicht (Feature #13)
    cols = st.columns(3)
    for idx, item in enumerate(items):
        with cols[idx % 3]:
            with st.container(border=True):
                st.image(item['image_url'], use_column_width=True)
                st.subheader(f"{item['label']}")
                st.write(f"📍 {item['location'] if item['location'] else 'Unbekannt'}")
                st.caption(f"Tags: {', '.join(item['tags'])}")
                if st.button("Kontakt aufnehmen", key=item['id']):
                    st.write("📧 Chat-Funktion (Feature #4) folgt...")
