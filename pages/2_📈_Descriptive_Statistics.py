# 2_📈_Descriptive_Statistics.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except Exception:
    px = None

# ------------------------------
# Configurazione pagina
# ------------------------------
st.set_page_config(page_title="Statistiche descrittive", layout="wide")

# ------------------------------
# Utility chiavi univoche
# ------------------------------
KEY_PREFIX = "ds"  # descriptive statistics

def k(name: str) -> str:
    return f"{KEY_PREFIX}_{name}"

def ss_get(name: str, default=None):
    return st.session_state.get(name, default)

def ss_set_default(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value

# ------------------------------
# Header
# ------------------------------
st.title("📈 Statistiche descrittive")
st.markdown(
    """
Questa sezione permette di esplorare i dati in modo **guidato**:
1) confermi che i dati siano disponibili;  
2) selezioni le variabili;  
3) ottenga tabelle e grafici automatici.
"""
)

# ==============================
# PASSO 0 · Recupero dati
# ==============================
st.subheader("Passo 0 · Dati disponibili")

df, found_key = None, None
for key in ["cleaned_df", "uploaded_df", "df_upload", "main_df", "dataset", "df"]:
    if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
        df, found_key = st.session_state[key], key
        break

if df is not None and not df.empty:
    st.success(f"Dataset pronto: {df.shape[0]} righe × {df.shape[1]} colonne")
    st.dataframe(df.head(15), use_container_width=True)
else:
    st.error("Nessun dataset trovato. Torni alla Pulizia Dati oppure carichi un file.")
    uploaded = st.file_uploader("Carichi qui un CSV/XLSX", type=["csv", "xlsx", "xls"], key=k("upload"))
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state["uploaded_df"] = df.copy()
            st.success("File caricato con successo.")
            st.dataframe(df.head(15), use_container_width=True)
        except Exception as e:
            st.error(f"Errore caricamento: {e}")
    if df is None or df.empty:
        st.stop()

# ==============================
# PASSO 1 · Selezione variabili
# ==============================
st.subheader("Passo 1 · Selezione variabili")

num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

c1, c2 = st.columns(2)
with c1:
    sel_num = st.multiselect("Variabili numeriche", options=num_vars, default=num_vars[:3], key=k("sel_num"))
with c2:
    sel_cat = st.multiselect("Variabili categoriali", options=cat_vars, default=cat_vars[:2], key=k("sel_cat"))

if not sel_num and not sel_cat:
    st.warning("Selezioni almeno una variabile (numerica o categoriale).")
    st.stop()

# ==============================
# PASSO 2 · Tabelle descrittive
# ==============================
st.subheader("Passo 2 · Tabelle descrittive")

if sel_num:
    st.markdown("**Statistiche per variabili numeriche**")
    desc_num = df[sel_num].describe().T
    desc_num["missing"] = df[sel_num].isna().sum()
    desc_num["unique"] = df[sel_num].nunique(dropna=True)
    st.dataframe(desc_num, use_container_width=True)

if sel_cat:
    st.markdown("**Distribuzione variabili categoriali**")
    for c in sel_cat:
        st.markdown(f"**{c}**")
        tab = df[c].value_counts(dropna=False)
        # Presentazione tabellare con etichette stringa (inclusi NaN come 'NaN')
        tab_df = tab.rename("Frequenza").reset_index().rename(columns={"index": c})
        tab_df[c] = tab_df[c].astype(str)
        st.dataframe(tab_df, use_container_width=True)

# ==============================
# PASSO 3 · Visualizzazioni
# ==============================
st.subheader("Passo 3 · Visualizzazioni")

if px is None:
    st.info("Plotly non disponibile nell'ambiente, impossibile generare grafici interattivi.")
else:
    g_num, g_cat = st.tabs(["Numeriche", "Categoriali"])

    with g_num:
        if sel_num:
            var_num = st.selectbox("Variabile numerica", options=sel_num, key=k("plot_num"))
            nbins = st.slider("Numero di bin (istogramma)", min_value=10, max_value=80, value=30, step=1, key=k("nbins"))
            fig = px.histogram(df, x=var_num, nbins=nbins, marginal="box", title=f"Distribuzione di {var_num}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessuna variabile numerica selezionata.")

    with g_cat:
        if sel_cat:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                var_cat = st.selectbox("Variabile categoriale", options=sel_cat, key=k("plot_cat"))
            with c2:
                top_n = st.number_input("Mostra Top N categorie (0 = tutte)", min_value=0, value=0, step=1, key=k("topn"))
            with c3:
                order_mode = st.selectbox("Ordine", ["Frequenza ↓", "Etichetta A→Z"], key=k("ordcat"))

            # Costruzione tabella frequenze con nomi espliciti e robusti
            vc = df[var_cat].value_counts(dropna=False)
            freq_df = vc.reset_index(name="count").rename(columns={"index": var_cat})
            # etichette come stringa (NaN -> 'NaN')
            freq_df[var_cat] = freq_df[var_cat].astype(str)

            if order_mode == "Frequenza ↓":
                freq_df = freq_df.sort_values("count", ascending=False)
            else:
                freq_df = freq_df.sort_values(var_cat, ascending=True)

            if top_n and top_n > 0:
                freq_df = freq_df.head(top_n)

            # Grafico a barre robusto
            fig = px.bar(
                freq_df,
                x=var_cat,
                y="count",
                title=f"Distribuzione di {var_cat}",
                text="count"
            )
            fig.update_layout(xaxis_title=var_cat, yaxis_title="Frequenza", bargap=0.2, height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessuna variabile categoriale selezionata.")

# ==============================
# PASSO 4 · Esportazione
# ==============================
st.subheader("Passo 4 · Salva ed esporta")

save_btn = st.button("Salva dataset per le analisi successive", key=k("save"))
if save_btn:
    st.session_state["descriptive_df"] = df.copy()
    st.success("Dataset salvato in `descriptive_df`.")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Scarica CSV",
    data=csv_bytes,
    file_name="dataset_descrittive.csv",
    mime="text/csv",
    key=k("download")
)
