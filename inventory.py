import streamlit as st
import pandas as pd
import os

# Konfiguration der Datei
DB_FILE = "inventar.csv"

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    else:
        # Standard-Struktur erstellen, falls Datei nicht existiert
        return pd.DataFrame(columns=["Produkt", "Kategorie", "Menge", "Preis"])

def save_data(df):
    df.to_csv(DB_FILE, index=False)

# App Titel
st.set_page_config(page_title="Mein Inventar-System", layout="wide")
st.title("📦 Inventar-Management")

# Daten laden
inventory = load_data()

# --- Sidebar: Neues Produkt hinzufügen ---
st.sidebar.header("Neuen Artikel hinzufügen")
with st.sidebar.form("add_form", clear_on_submit=True):
    name = st.text_input("Produktname")
    cat = st.selectbox("Kategorie", ["Elektronik", "Büro", "Lager", "Sonstiges"])
    qty = st.number_input("Menge", min_value=0, step=1)
    price = st.number_input("Preis (€)", min_value=0.0, step=0.01)
    submit = st.form_submit_button("Hinzufügen")

    if submit and name:
        new_item = pd.DataFrame([[name, cat, qty, price]], columns=inventory.columns)
        inventory = pd.concat([inventory, new_item], ignore_index=True)
        save_data(inventory)
        st.success(f"{name} hinzugefügt!")

# --- Hauptbereich: Inventar anzeigen & bearbeiten ---
st.subheader("Aktueller Lagerbestand")

if not inventory.empty:
    # Suchfunktion
    search = st.text_input("Suchen...", "")
    filtered_df = inventory[inventory['Produkt'].str.contains(search, case=False)]

    # Anzeige der Tabelle
    # 'num_rows="dynamic"' erlaubt das direkte Löschen/Hinzufügen in der UI
    edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)

    if st.button("Änderungen speichern"):
        save_data(edited_df)
        st.success("Inventar aktualisiert!")
else:
    st.info("Das Inventar ist noch leer.")

# --- Statistiken ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.metric("Gesamtanzahl Artikel", int(inventory["Menge"].sum()))
with col2:
    wert = (inventory["Menge"] * inventory["Preis"]).sum()
    st.metric("Gesamtwert des Lagers", f"{wert:,.2f} €")