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
# Navigazione agli step successivi
# ---------------------------------
PAGES = [
    ("pages/1_🧹_Data_Cleaning.py",        "Vai a: Cleaning",               "🧹"),
    ("pages/2_📈_Descriptive_Statistics.py","Vai a: Descrittive",            "📈"),
    ("pages/3_📊_Explore_Distributions.py", "Vai a: Esplora distribuzioni",  "📊"),
    ("pages/4_🔍_Assumption_Checks.py",     "Vai a: Assumption checks",      "🔍"),
    ("pages/5_🧪_Statistical_Tests.py",     "Vai a: Test statistici",        "🧪"),
    ("pages/6_🔗_Correlation.py",           "Vai a: Correlazioni",           "🔗"),
    # In futuro: Results Summary
    # ("pages/7_🧾_Results_Summary.py",     "Vai a: Results Summary",        "🧾"),
]

st.divider()
st.subheader("Navigazione")

col_nav1, col_nav2, col_nav3 = st.columns(3)
cols = [col_nav1, col_nav2, col_nav3]

i = 0
for page_path, label, icon in PAGES:
    if Path(page_path).exists():
        with cols[i % 3]:
            st.page_link(page_path, label=f"{label}", icon=icon)
        i += 1
