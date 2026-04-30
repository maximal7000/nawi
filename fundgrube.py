import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client, Client
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import uuid
import io
import os
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

# --- 3. KI MODELL FUNKTIONEN (Fashion-MNIST von Zalando) ---
FASHION_LABELS_DE = [
    "T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
    "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefelette",
]

WEIGHTS_PATH = "fashion_cnn.pt"

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def _train_model(model, epochs=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    progress = st.progress(0.0, text="Trainiere Fashion-MNIST CNN...")
    total_steps = epochs * len(loader)
    step = 0
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                progress.progress(step / total_steps, text=f"Trainiere... ({step}/{total_steps})")
    progress.empty()

@st.cache_resource(show_spinner="Lade Fashion-MNIST Modell...")
def load_model():
    model = FashionCNN()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    else:
        _train_model(model, epochs=2)
        torch.save(model.state_dict(), WEIGHTS_PATH)
    model.eval()
    return model

def predict(image_data, model, top_k=3):
    """Klassifiziert Bild gegen die 10 Fashion-MNIST Klassen. Gibt Top-K Vorhersagen zurück."""
    img = ImageOps.grayscale(image_data)
    # Fashion-MNIST: helle Objekte auf dunklem Hintergrund. Reale Fotos sind meist umgekehrt.
    arr_check = np.asarray(img, dtype=np.float32)
    if arr_check.mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    predictions = [{"label": FASHION_LABELS_DE[int(i)], "confidence": float(probs[int(i)])} for i in top_idx]
    return predictions

def check_for_matches(new_label, current_type):
    target_type = "search" if current_type == "found" else "found"
    try:
        matches = supabase.table("items").select("*").eq("label", new_label).eq("type", target_type).execute()
        return matches.data
    except:
        return []

# --- 4. NAVIGATION ---
st.sidebar.title("🔍 KI-Fundbüro")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📤 Etwas melden (KI)", "📦 Datenbank durchsuchen"])

model = load_model()

# --- MODUS: HOME ---
if menu == "🏠 Home":
    st.title("Willkommen bei der KI-Fundgrube")
    st.write("Verlorene Kleidungsstücke erkennen – mit Fashion-MNIST (Zalando) und einem eigenen CNN.")

    col1, col2 = st.columns(2)
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
    st.caption("Erkennt: T-Shirt, Hose, Pullover, Kleid, Mantel, Sandale, Hemd, Sneaker, Tasche, Stiefelette.")

    mode = st.radio("Was möchtest du tun?", ["Ich habe etwas gefunden", "Ich vermisse etwas"], horizontal=True)
    db_type = "found" if "gefunden" in mode else "search"

    file = st.file_uploader("Bild des Gegenstands hochladen", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", width=300)

        if st.button("KI-Analyse starten (Fashion-MNIST)"):
            with st.spinner("Analysiere Bild..."):
                predictions = predict(image, model, top_k=3)
                st.session_state["predictions"] = predictions
                st.session_state["detected_label"] = predictions[0]["label"]

        if "predictions" in st.session_state:
            st.divider()
            preds = st.session_state["predictions"]
            options = [f"{p['label']} ({round(p['confidence']*100,1)}%)" for p in preds]
            selected = st.selectbox("Top-Vorhersagen — welche passt?", options, index=0)
            st.session_state["detected_label"] = preds[options.index(selected)]["label"]
            st.success(f"Ausgewähltes Objekt: **{st.session_state['detected_label']}**")

        if st.session_state.get("detected_label"):
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                location = st.text_input("📍 Ort (z.B. Hamburg, Altona)")
                reward = st.number_input("💰 Finderlohn/Belohnung (€)", min_value=0, value=0)
            with col_b:
                tags = st.text_input("🏷 Tags (kommagetrennt)", value=f"{st.session_state['detected_label']}, {datetime.now().year}")

            if st.button("Eintrag speichern"):
                with st.spinner("Wird gespeichert..."):
                    file_name = f"{db_type}/{uuid.uuid4()}.jpg"
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")

                    try:
                        img_byte_arr.seek(0)
                        supabase.storage.from_("images").upload(
                            path=file_name,
                            file=img_byte_arr.getvalue(),
                            file_options={"content-type": "image/jpeg"},
                        )
                        img_url = supabase.storage.from_("images").get_public_url(file_name)

                        new_item = {
                            "label": st.session_state["detected_label"],
                            "tags": [t.strip() for t in tags.split(",")],
                            "image_url": img_url,
                            "type": db_type,
                            "location": location,
                            "reward": reward,
                        }
                        supabase.table("items").insert(new_item).execute()

                        st.success("Erfolgreich gespeichert!")

                        matches = check_for_matches(st.session_state["detected_label"], db_type)
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
                items = [i for i in items if search_query.lower() in i["label"].lower() or search_query.lower() in str(i["tags"]).lower()]

            cols = st.columns(3)
            for idx, item in enumerate(items):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(item["image_url"], use_column_width=True)
                        st.subheader(item["label"])
                        st.write(f"**Typ:** {'Gefunden' if item['type'] == 'found' else 'Vermisst'}")
                        st.write(f"📍 {item['location'] if item['location'] else 'Unbekannt'}")
                        if item["reward"] > 0:
                            st.write(f"💰 {item['reward']}€ Belohnung")
                        st.caption(f"Tags: {', '.join(item['tags'])}")
        else:
            st.info("Noch keine Einträge vorhanden.")
    except Exception as e:
        st.error("Daten konnten nicht geladen werden. Existiert die Tabelle 'items'?")
