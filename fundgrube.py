import streamlit as st
from PIL import Image
import numpy as np
from supabase import create_client, Client
from ultralytics import YOLO
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
# Übersetzungstabelle für gängige COCO-Klassen ins Deutsche
COCO_DE = {
    "person": "Person", "bicycle": "Fahrrad", "car": "Auto", "motorcycle": "Motorrad",
    "airplane": "Flugzeug", "bus": "Bus", "train": "Zug", "truck": "LKW", "boat": "Boot",
    "traffic light": "Ampel", "fire hydrant": "Hydrant", "stop sign": "Stoppschild",
    "parking meter": "Parkuhr", "bench": "Bank", "bird": "Vogel", "cat": "Katze",
    "dog": "Hund", "horse": "Pferd", "sheep": "Schaf", "cow": "Kuh", "elephant": "Elefant",
    "bear": "Bär", "zebra": "Zebra", "giraffe": "Giraffe", "backpack": "Rucksack",
    "umbrella": "Regenschirm", "handbag": "Handtasche", "tie": "Krawatte",
    "suitcase": "Koffer", "frisbee": "Frisbee", "skis": "Ski", "snowboard": "Snowboard",
    "sports ball": "Ball", "kite": "Drachen", "baseball bat": "Baseballschläger",
    "baseball glove": "Baseballhandschuh", "skateboard": "Skateboard",
    "surfboard": "Surfbrett", "tennis racket": "Tennisschläger", "bottle": "Flasche",
    "wine glass": "Weinglas", "cup": "Tasse", "fork": "Gabel", "knife": "Messer",
    "spoon": "Löffel", "bowl": "Schüssel", "banana": "Banane", "apple": "Apfel",
    "sandwich": "Sandwich", "orange": "Orange", "broccoli": "Brokkoli",
    "carrot": "Karotte", "hot dog": "Hotdog", "pizza": "Pizza", "donut": "Donut",
    "cake": "Kuchen", "chair": "Stuhl", "couch": "Sofa", "potted plant": "Topfpflanze",
    "bed": "Bett", "dining table": "Esstisch", "toilet": "Toilette", "tv": "Fernseher",
    "laptop": "Laptop", "mouse": "Maus", "remote": "Fernbedienung", "keyboard": "Tastatur",
    "cell phone": "Handy", "microwave": "Mikrowelle", "oven": "Ofen", "toaster": "Toaster",
    "sink": "Spüle", "refrigerator": "Kühlschrank", "book": "Buch", "clock": "Uhr",
    "vase": "Vase", "scissors": "Schere", "teddy bear": "Teddybär", "hair drier": "Föhn",
    "toothbrush": "Zahnbürste",
}

@st.cache_resource
def load_model():
    # Lädt das YOLOv8-Nano-Modell (wird beim ersten Start automatisch heruntergeladen)
    return YOLO('yolov8n.pt')

def predict(image_data, model, conf_threshold=0.25):
    """Führt YOLOv8-Erkennung durch und gibt (annotiertes Bild, Liste an Detections) zurück."""
    img_array = np.asarray(image_data)
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]

    # Annotiertes Bild (BGR -> RGB)
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        en_label = result.names[cls_id]
        de_label = COCO_DE.get(en_label, en_label)
        detections.append({"label": de_label, "label_en": en_label, "confidence": conf})

    # Nach Konfidenz sortieren (höchste zuerst)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return annotated_image, detections

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

# Laden des YOLOv8-Modells
model = load_model()

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
    
    conf_threshold = st.slider("Mindest-Konfidenz für Erkennung", 0.10, 0.90, 0.25, 0.05)

    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Hochgeladenes Bild", width=300)

        if st.button("KI-Analyse starten (YOLOv8)"):
            with st.spinner("YOLOv8 analysiert das Bild..."):
                annotated_image, detections = predict(image, model, conf_threshold)
                st.session_state['annotated_image'] = annotated_image
                st.session_state['detections'] = detections
                if detections:
                    st.session_state['detected_label'] = detections[0]['label']
                else:
                    st.session_state.pop('detected_label', None)

        if 'detections' in st.session_state:
            st.divider()
            st.image(st.session_state['annotated_image'], caption="YOLOv8-Erkennung", use_column_width=True)

            detections = st.session_state['detections']
            if not detections:
                st.warning("Es wurden keine Objekte erkannt. Versuche es mit einem anderen Bild oder einer niedrigeren Konfidenzschwelle.")
            else:
                options = [f"{d['label']} ({round(d['confidence']*100,1)}%)" for d in detections]
                selected = st.selectbox("Welches Objekt möchtest du melden?", options, index=0)
                st.session_state['detected_label'] = detections[options.index(selected)]['label']
                st.success(f"Ausgewähltes Objekt: **{st.session_state['detected_label']}**")

        if st.session_state.get('detected_label'):
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
