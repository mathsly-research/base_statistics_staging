# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# -----------------------------
# Init & dataset check
# -----------------------------
init_state()
st.title("🧹 Step 1 — Gestione dati mancanti e filtri")

# Manteniamo sempre copia originale
if st.session_state.df_original is None and st.session_state.df is not None:
    st.session_state.df_original = st.session_state.df.copy()
if st.session_state.df_working is None and st.session_state.df is not None:
    st.session_state.df_working = st.session_state.df.copy()

if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

# -----------------------------
# Selettore dataset attivo
# -----------------------------
choice = st.radio(
    "Seleziona dataset attivo:",
    ["Originale", "Modificato (working)"],
    horizontal=True
)

if choice == "Originale":
    df = st.session_state.df_original.copy()
else:
    df = st.session_state.df_working.copy()

st.info(f"Dataset attivo: **{choice}** — {df.shape[0]} righe × {df.shape[1]} colonne")

# -----------------------------
# Diagnostica missing values
# -----------------------------
st.subheader("🔎 Missing values per variabile")
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
summary = pd.DataFrame({
    "Missing": missing,
    "% Missing": missing_pct.round(1)
})
st.dataframe(summary, use_container_width=True)

# -----------------------------
# Strategie di gestione missing
# -----------------------------
st.subheader("⚙️ Strategie di gestione")
options = {
    "Nessuna azione": None,
    "Elimina righe con missing (sconsigliato)": "drop_rows",
    "Elimina variabile (colonna) (sconsigliato)": "drop_col",
    "Sostituisci con media (solo numeriche)": "mean",
    "Sostituisci con mediana (solo numeriche)": "median",
    "Sostituisci con moda": "mode"
}

strategy = {}
for col in df.columns:
    strategy[col] = st.selectbox(
        f"{col}:",
        list(options.keys()),
        index=0,
        key=f"strategy_{col}"
    )
    if "sconsigliato" in strategy[col]:
        st.warning(f"⚠️ {col}: l'eliminazione può introdurre bias. Usare solo se strettamente necessario.")

if st.button("➡️ Applica strategie di gestione"):
    for col, choice in strategy.items():
        if options[choice] == "drop_rows":
            df = df.dropna(subset=[col])
        elif options[choice] == "drop_col":
            df = df.drop(columns=[col])
        elif options[choice] == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        elif options[choice] == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        elif options[choice] == "mode":
            if not df[col].dropna().empty:
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)

    st.session_state.df_working = df
    st.success("Strategie applicate al dataset modificato (working).")
    st.warning("⚠️ I risultati precedenti potrebbero non essere più coerenti: ricalcolarli se necessario.")

# -----------------------------
# Filtri interattivi
# -----------------------------
st.subheader("🔎 Filtri sui dati")

filters = {}
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = float(df[col].min()), float(df[col].max())
        sel_min, sel_max = st.slider(
            f"Filtro per {col}:",
            min_value=min_val, max_value=max_val,
            value=(min_val, max_val)
        )
        filters[col] = (sel_min, sel_max)
    else:
        unique_vals = df[col].dropna().unique().tolist()
        sel_vals = st.multiselect(f"Filtro per {col}:", unique_vals, default=unique_vals)
        filters[col] = sel_vals

if st.button("➡️ Applica filtri"):
    for col, rule in filters.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            df = df[(df[col] >= rule[0]) & (df[col] <= rule[1])]
        else:
            df = df[df[col].isin(rule)]
    st.session_state.df_working = df
    st.success(f"Filtri applicati al dataset modificato (working): {df.shape[0]} righe rimanenti.")
    st.warning("⚠️ I risultati precedenti potrebbero non essere più coerenti: ricalcolarli se necessario.")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Guida"):
    st.markdown("""
**Gestione dei dati mancanti:**  
- *Elimina righe / colonne*: **sconsigliato** perché può introdurre bias o perdita di informazione.  
- *Imputazione (media/mediana/moda)*: mantiene il dataset completo, ma attenzione a non introdurre distorsioni.  

**Dataset attivi:**  
- **Originale**: resta invariato e serve come riferimento.  
- **Modificato (working)**: contiene le modifiche applicate (imputazioni, filtri, ecc.).  

⚠️ Se cambi dataset, i risultati già salvati potrebbero non essere più coerenti.
""")
