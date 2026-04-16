import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client, Client
import uuid
import io
import pandas as pd
from datetime import datetime

# --- 1. SEITEN KONFIGURATION & STYLING ---
st.set_page_config(page_title="KI Fundgrube Pro", page_icon="🔍", layout="wide")

# Erzwinge Darkmode-Optik via CSS
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #464b5d; }
        .stTextInput>div>div>input { background-color: #262730; color: white; }
        [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SUPABASE SETUP (NUR ÜBER SECRETS) ---
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")
        st.stop()

supabase = init_connection()

# --- 3. KI MODELL FUNKTIONEN ---
@st.cache_resource
def load_model():
    # Lädt dein Teachable Machine Modell
    return tf.keras.models.load_model('keras_model.h5', compile=False)

def predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array
    return model.predict(data)

def check_for_matches(new_label, current_type):
    # Sucht nach dem Gegenteil (Suche vs. Fund)
    target_type = "search" if current_type == "found" else "found"
    try:
        matches = supabase.table("items").select("*").eq("label", new_label).eq("type", target_type).execute()
        return matches.data
    except:
        return []

# --- 4. NAVIGATION ---
st.sidebar.title("🔍 KI-Fundbüro")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📤 Etwas melden (KI)", "📦 Datenbank durchsuchen"])

# Laden von Modell und Labels
model = load_model()
try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ["0 Unbekannt"]

# --- MODUS: HOME ---
if menu == "🏠 Home":
    st.title("Willkommen bei der KI-Fundgrube")
    st.write("Verlorene Gegenstände finden – mit künstlicher Intelligenz.")
    
    col1, col2 = st.columns(2)
    # Statistiken direkt aus der DB
    try:
        all_items = supabase.table("items").select("id", count="exact").execute()
        count = all_items.count if all_items.count else 0
        col1.metric("Registrierte Objekte", count)
        col2.metric("System-Status", "Aktiv")
    except:
        col1.info("Datenbank bereit für ersten Eintrag.")

# --- MODUS: MELDEN ---
elif menu == "📤 Etwas melden (KI)":
    st.header("Objekt mit KI erfassen")
    
    mode = st.radio("Was möchtest du tun?", ["Ich habe etwas gefunden", "Ich vermisse etwas"], horizontal=True)
    db_type = "found" if "gefunden" in mode else "search"
    
    file = st.file_uploader("Bild des Gegenstands hochladen", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Hochgeladenes Bild", width=300)
        
        if st.button("KI-Analyse starten"):
            with st.spinner("Analysiere..."):
                prediction = predict(image, model)
                idx = np.argmax(prediction)
                # Entfernt die Nummer am Anfang des Labels (z.B. "0 Schlüssel" -> "Schlüssel")
                label = class_names[idx].split(' ', 1)[1] if ' ' in class_names[idx] else class_names[idx]
                conf = prediction[0][idx]
                
                st.session_state['detected_label'] = label
                st.success(f"Objekt erkannt: **{label}** ({round(conf*100,1)}%)")

        if 'detected_label' in st.session_state:
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                location = st.text_input("📍 Ort (z.B. Hamburg, Altona)")
                reward = st.number_input("💰 Finderlohn/Belohnung (€)", min_value=0, value=0)
            with col_b:
                tags = st.text_input("🏷 Tags (kommagetrennt)", value=f"{st.session_state['detected_label']}, {datetime.now().year}")

            if st.button("Eintrag speichern"):
                with st.spinner("Wird gespeichert..."):
                    # 1. Bild-Upload in Supabase Storage
                    file_name = f"{db_type}/{uuid.uuid4()}.jpg"
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    
                    try:
                        img_byte_arr.seek(0)
                        supabase.storage.from_("images").upload(
                            path=file_name,
                            file=img_byte_arr.getvalue(),
                            file_options={"content-type": "image/jpeg"}
                        )
                        img_url = supabase.storage.from_("images").get_public_url(file_name)

                        # 2. Datenbank-Eintrag
                        new_item = {
                            "label": st.session_state['detected_label'],
                            "tags": [t.strip() for t in tags.split(",")],
                            "image_url": img_url,
                            "type": db_type,
                            "location": location,
                            "reward": reward
                        }
                        supabase.table("items").insert(new_item).execute()
                        
                        st.success("Erfolgreich gespeichert!")
                        
                        # 3. Matching Check
                        matches = check_for_matches(st.session_state['detected_label'], db_type)
                        if matches:
                            st.balloons()
                            st.info(f"🎉 Wir haben {len(matches)} potenzielle Treffer in der Datenbank gefunden!")
                    except Exception as e:
                        st.error(f"Fehler beim Speichern: {e}")

# --- MODUS: DURCHSUCHEN ---
elif menu == "📦 Datenbank durchsuchen":
    st.header("Aktuelle Funde & Gesuche")
    
    try:
        res = supabase.table("items").select("*").order("created_at", desc=True).execute()
        items = res.data

        if items:
            search_query = st.text_input("🔍 Filtern nach Name oder Tag...")
            if search_query:
                items = [i for i in items if search_query.lower() in i['label'].lower() or search_query.lower() in str(i['tags']).lower()]

            cols = st.columns(3)
            for idx, item in enumerate(items):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(item['image_url'], use_column_width=True)
                        st.subheader(item['label'])
                        st.write(f"**Typ:** {'Gefunden' if item['type'] == 'found' else 'Vermisst'}")
                        st.write(f"📍 {item['location'] if item['location'] else 'Unbekannt'}")
                        if item['reward'] > 0:
                            st.write(f"💰 {item['reward']}€ Belohnung")
                        st.caption(f"Tags: {', '.join(item['tags'])}")
        else:
            st.info("Noch keine Einträge vorhanden.")
    except Exception as e:
        st.error("Daten konnten nicht geladen werden. Existiert die Tabelle 'items'?")
