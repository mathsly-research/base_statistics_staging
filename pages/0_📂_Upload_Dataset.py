# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from pathlib import Path

from core.state import init_state

# ---------------------------------
# Init
# ---------------------------------
init_state()
st.title("📂 Step 0 — Upload Dataset")

# ---------------------------------
# Upload file
# ---------------------------------
uploaded_file = st.file_uploader("Carica il tuo dataset (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Salvataggio nel session state
        st.session_state.df = df.copy()
        st.session_state.df_original = df.copy()   # sempre intatto
        st.session_state.df_working = df.copy()    # usato per cleaning/filtri

        st.success(f"✅ Dataset caricato correttamente: {df.shape[0]} righe × {df.shape[1]} colonne")

        # Preview
        st.subheader("Anteprima dataset")
        st.dataframe(df.head(), use_container_width=True)

        # Info variabili
        st.subheader("📋 Informazioni sulle variabili")
        info = pd.DataFrame({
            "Colonna": df.columns,
            "Tipo": [str(df[c].dtype) for c in df.columns],
            "Valori unici": [df[c].nunique() for c in df.columns],
            "Missing": [df[c].isna().sum() for c in df.columns]
        })
        st.dataframe(info, use_container_width=True)

    except Exception as e:
        st.error(f"Errore durante la lettura del file: {e}")

else:
    st.info("Carica un file per iniziare.")

# ---------------------------------
# Navigazione agli step successivi (stile dashboard colorata)
# ---------------------------------
PAGES = [
    ("pages/1_🧹_Data_Cleaning.py",        "🧹", "Cleaning",         "Gestione missing values e filtri", "#e6f7ff"),
    ("pages/2_📈_Descriptive_Statistics.py","📈", "Descrittive",      "Statistiche di base e riepilogo variabili", "#fff5e6"),
    ("pages/3_📊_Explore_Distributions.py", "📊", "Distribuzioni",    "Istogrammi, boxplot e violino", "#f9e6ff"),
    ("pages/4_🔍_Assumption_Checks.py",     "🔍", "Assunzioni",       "Verifica normalità e omoscedasticità", "#e6ffe6"),
    ("pages/5_🧪_Statistical_Tests.py",     "🧪", "Test statistici",  "Confronti parametrici e non parametrici", "#fff0f0"),
    ("pages/6_🔗_Correlation.py",           "🔗", "Correlazioni",     "Relazioni tra variabili e heatmap", "#f0f5ff"),
    # Futuro: Results Summary
    # ("pages/7_🧾_Results_Summary.py",     "🧾", "Report finale",   "Sintesi dei risultati ed esportazione", "#f5f5f5")
]

st.divider()
st.subheader("🚀 Navigazione rapida agli step")

cols = st.columns(2)  # due colonne, stile dashboard
i = 0
for page_path, icon, title, desc, color in PAGES:
    if Path(page_path).exists():
        with cols[i % 2]:
            st.markdown(
                f"""
                <div style="border-radius:15px; padding:20px; margin-bottom:20px; 
                            background-color:{color}; text-align:center; 
                            box-shadow:2px 2px 10px rgba(0,0,0,0.08);">
                    <div style="font-size:40px; margin-bottom:10px;">{icon}</div>
                    <h3 style="margin-bottom:5px; color:#333;">{title}</h3>
                    <p style="margin-top:0; margin-bottom:15px; color:#555; font-size:0.9em;">{desc}</p>
                    <a href='/{page_path}' target='_self' style='text-decoration:none;'>
                        <button style="background-color:#4CAF50; color:white; padding:10px 20px; 
                                       border:none; border-radius:10px; cursor:pointer; font-size:0.9em;">
                            Vai →
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        i += 1


