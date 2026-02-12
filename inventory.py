import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

# Konfiguration der Seite
st.set_page_config(page_title="Inventar-Manager", layout="wide")

DB_FILE = "inventar.json"

# --- DATEN-FUNKTIONEN ---

def lade_daten():
    """Lädt die Daten aus der JSON-Datei."""
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def speichere_daten(daten):
    """Speichert die Daten in der JSON-Datei."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(daten, f, indent=4, ensure_ascii=False)

# --- HAUPT-APP ---

def main():
    st.title("📦 Mein Inventar-System")
    
    # Daten initial laden
    if 'inventar' not in st.session_state:
        st.session_state.inventar = lade_daten()

    # Sidebar für Statistiken
    st.sidebar.header("Statistik")
    st.sidebar.info(f"Gesamtanzahl Artikel: {len(st.session_state.inventar)}")

    # Tabs für die CRUD-Operationen
    tab1, tab2, tab3 = st.tabs(["📋 Übersicht & Suche", "➕ Neu anlegen", "⚙️ Verwalten"])

    # --- TAB 1: ÜBERSICHT ---
    with tab1:
        st.subheader("Aktueller Bestand")
        if not st.session_state.inventar:
            st.write("Das Inventar ist noch leer.")
        else:
            df = pd.DataFrame(st.session_state.inventar)
            
            # Suchfunktion
            suche = st.text_input("Suche nach Name oder Kategorie", "")
            if suche:
                df = df[df['Name'].str.contains(suche, case=False) | 
                        df['Kategorie'].str.contains(suche, case=False)]
            
            st.dataframe(df, use_container_width=True, hide_index=True)

    # --- TAB 2: NEU ANLEGEN (CREATE) ---
    with tab2:
        st.subheader("Neuen Gegenstand hinzufügen")
        with st.form("add_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name des Gegenstands")
                kat = st.selectbox("Kategorie", ["Werkzeug", "Elektronik", "Büro", "Möbel", "Sonstiges"])
            with col2:
                menge = st.number_input("Menge", min_value=0, step=1)
                ort = st.text_input("Standort (Lagerfach, Raum)")
            
            kommentar = st.text_area("Kommentar")
            submit = st.form_submit_button("Speichern")

            if submit:
                if name:
                    neuer_artikel = {
                        "ID": datetime.now().strftime("%Y%m%d%H%M%S"), # Zeitstempel als einfache ID
                        "Name": name,
                        "Kategorie": kat,
                        "Menge": menge,
                        "Standort": ort,
                        "Kommentar": kommentar
                    }
                    st.session_state.inventar.append(neuer_artikel)
                    speichere_daten(st.session_state.inventar)
                    st.success(f"'{name}' wurde hinzugefügt!")
                    st.rerun()
                else:
                    st.error("Bitte gib mindestens einen Namen an.")

    # --- TAB 3: VERWALTEN (UPDATE & DELETE) ---
    with tab3:
        st.subheader("Einträge bearbeiten oder löschen")
        if not st.session_state.inventar:
            st.write("Keine Daten zum Verwalten vorhanden.")
        else:
            artikelliste = [item["Name"] for item in st.session_state.inventar]
            auswahl = st.selectbox("Wähle einen Artikel zum Bearbeiten", artikelliste)
            
            # Den gewählten Artikel finden
            artikel_index = next((index for (index, d) in enumerate(st.session_state.inventar) if d["Name"] == auswahl), None)
            artikel = st.session_state.inventar[artikel_index]

            with st.expander(f"Details für {artikel['Name']}"):
                u_name = st.text_input("Name", value=artikel['Name'])
                u_kat = st.text_input("Kategorie", value=artikel['Kategorie'])
                u_menge = st.number_input("Menge", value=artikel['Menge'], min_value=0)
                u_ort = st.text_input("Standort", value=artikel['Standort'])
                u_komm = st.text_area("Kommentar ", value=artikel['Kommentar'], key="edit_kom")
                
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("Änderungen speichern", type="primary"):
                    st.session_state.inventar[artikel_index] = {
                        "ID": artikel['ID'], "Name": u_name, "Kategorie": u_kat,
                        "Menge": u_menge, "Standort": u_ort, "Kommentar": u_komm
                    }
                    speichere_daten(st.session_state.inventar)
                    st.success("Aktualisiert!")
                    st.rerun()
                
                if col_btn2.button("Löschen", type="secondary"):
                    st.session_state.inventar.pop(artikel_index)
                    speichere_daten(st.session_state.inventar)
                    st.warning("Artikel gelöscht.")
                    st.rerun()

if __name__ == "__main__":
    main()
