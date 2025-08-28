# pages/2_📈_Descriptive_Statistics.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except Exception:
    px = None

# ──────────────────────────────────────────────────────────────────────────────
# Data Store (preferito) + fallback
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized(); df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Esegua Upload/Pulizia dati."); st.stop()
        return df
    def stamp_meta():
        ensure_initialized()
        meta = st.session_state["ds_meta"]; ver = meta.get("version", 0); src = meta.get("source") or "-"
        ts = meta.get("updated_at"); when = "-"
        if ts:
            from datetime import datetime; when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Statistiche descrittive", layout="wide")
try:
    from nav import sidebar; sidebar()
except Exception:
    pass

KEY = "ds"
def k(name: str) -> str: return f"{KEY}_{name}"

st.title("📈 Statistiche descrittive")
ensure_initialized()

# 🔴 Dataset attivo (aggiornato dal Cleaning)
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# PASSO 1 · Selezione variabili
st.subheader("Passo 1 · Selezione variabili")
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
c1, c2 = st.columns(2)
with c1: sel_num = st.multiselect("Variabili numeriche", options=num_vars, default=num_vars[:3], key=k("num"))
with c2: sel_cat = st.multiselect("Variabili categoriali", options=cat_vars, default=cat_vars[:2], key=k("cat"))
if not sel_num and not sel_cat:
    st.warning("Selezioni almeno una variabile."); st.stop()

# PASSO 2 · Tabelle
st.subheader("Passo 2 · Tabelle descrittive")
if sel_num:
    st.markdown("**Statistiche numeriche**")
    desc = df[sel_num].describe().T
    desc["missing"] = df[sel_num].isna().sum()
    desc["unique"]  = df[sel_num].nunique()
    st.dataframe(desc, use_container_width=True)
if sel_cat:
    st.markdown("**Distribuzioni categoriali**")
    for c in sel_cat:
        tab = df[c].value_counts(dropna=False).reset_index(name="count").rename(columns={"index": c})
        tab[c] = tab[c].astype(str)
        st.markdown(f"**{c}**")
        st.dataframe(tab, use_container_width=True)

# PASSO 3 · Grafici
st.subheader("Passo 3 · Visualizzazioni")
if px is None:
    st.info("Plotly non disponibile nell'ambiente.")
else:
    t1, t2 = st.tabs(["Numeriche", "Categoriali"])
    with t1:
        if sel_num:
            var = st.selectbox("Variabile numerica", options=sel_num, key=k("plot_num"))
            nb = st.slider("Bin istogramma", 10, 80, 30, 1, key=k("bins"))
            fig = px.histogram(df, x=var, nbins=nb, marginal="box", title=f"Distribuzione di {var}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessuna variabile numerica selezionata.")
    with t2:
        if sel_cat:
            var = st.selectbox("Variabile categoriale", options=sel_cat, key=k("plot_cat"))
            topn = st.number_input("Top N (0 = tutte)", min_value=0, value=0, step=1, key=k("topn"))
            order = st.selectbox("Ordina per", ["Frequenza ↓","Etichetta A→Z"], key=k("ord"))
            freq = df[var].value_counts(dropna=False).reset_index(name="count").rename(columns={"index": var})
            freq[var] = freq[var].astype(str)
            if order == "Frequenza ↓": freq = freq.sort_values("count", ascending=False)
            else: freq = freq.sort_values(var)
            if topn: freq = freq.head(topn)
            fig = px.bar(freq, x=var, y="count", text="count", title=f"Distribuzione di {var}")
            fig.update_layout(yaxis_title="Frequenza", height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessuna variabile categoriale selezionata.")

# PASSO 4 · Export (opzionale)
st.subheader("Passo 4 · Esportazione")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica CSV corrente", data=csv_bytes, file_name="dataset_corrente.csv",
                   mime="text/csv", key=k("dl"))
