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
# Data store (preferito) + fallback
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
        meta = st.session_state["ds_meta"]; ver = meta.get("version", 0); src = meta.get("source") or "-"
        ts = meta.get("updated_at"); when = "-"
        if ts:
            from datetime import datetime; when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Statistiche descrittive", layout="wide")
try:
    from nav import sidebar; sidebar()
except Exception:
    pass

KEY = "ds"
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 Statistiche descrittive")
ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# ──────────────────────────────────────────────────────────────────────────────
# Passo 1 · Selezione variabili
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 1 · Selezione variabili")

num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

c1, c2 = st.columns(2)
with c1:
    sel_num = st.multiselect("Variabili numeriche (per tabelle e grafici)", options=num_vars, default=num_vars[:3], key=k("sel_num"))
with c2:
    sel_cat = st.multiselect("Variabili categoriali", options=cat_vars, default=cat_vars[:2], key=k("sel_cat"))

if not sel_num and not sel_cat:
    st.warning("Selezioni almeno una variabile (numerica o categoriale).")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Passo 2 · Tabelle descrittive
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Tabelle descrittive")

if sel_num:
    st.markdown("**Statistiche per variabili numeriche**")
    desc_num = df[sel_num].describe().T
    desc_num["missing"] = df[sel_num].isna().sum()
    desc_num["unique"] = df[sel_num].nunique()
    st.dataframe(desc_num, use_container_width=True)

if sel_cat:
    st.markdown("**Distribuzione variabili categoriali**")
    for c in sel_cat:
        vc = df[c].value_counts(dropna=False)
        tab = vc.reset_index(name="count").rename(columns={"index": c})
        tab[c] = tab[c].astype(str)
        tab["percent"] = (tab["count"] / len(df) * 100).round(2)
        st.markdown(f"**{c}**")
        st.dataframe(tab, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Passo 3 · Visualizzazioni
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 3 · Visualizzazioni")

if px is None:
    st.info("Plotly non disponibile nell'ambiente.")
else:
    tabs = st.tabs(["🔢 Numeriche", "🔠 Categoriali"])

    # ==============================
    # NUMERICHE (menu a tendina)
    # ==============================
    with tabs[0]:
        if sel_num:
            active_num = st.selectbox("Variabile numerica da visualizzare", options=sel_num, key=k("active_num"))

            hue = st.selectbox("Colore per sottogruppo (opzionale)", options=["(nessuno)"] + cat_vars, key=k("hue"))
            hue = None if hue == "(nessuno)" else hue

            nbins = st.slider("Numero di bin (istogramma)", 10, 100, 30, 5, key=k("nbins"))
            style = st.selectbox("Secondo grafico", ["Boxplot", "Violin"], key=k("style"))

            left, right = st.columns(2, vertical_alignment="top")

            with left:
                fig_hist = px.histogram(df, x=active_num, color=hue, nbins=nbins, template="simple_white")
                fig_hist.update_layout(title=f"Istogramma di {active_num}", height=400, font=dict(size=14))
                st.plotly_chart(fig_hist, use_container_width=True)

            with right:
                if style == "Boxplot":
                    fig_box = px.box(df, y=active_num, color=hue, points="all", template="simple_white")
                    fig_box.update_layout(title=f"Boxplot di {active_num}", height=400, font=dict(size=14))
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_violin = px.violin(df, y=active_num, color=hue, box=True, points="all", template="simple_white")
                    fig_violin.update_layout(title=f"Violin plot di {active_num}", height=400, font=dict(size=14))
                    st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Nessuna variabile numerica selezionata.")

    # ==============================
    # CATEGORIALI
    # ==============================
    with tabs[1]:
        if sel_cat:
            var_cat = st.selectbox("Variabile categoriale da visualizzare", options=sel_cat, key=k("plot_cat"))
            show_mode = st.selectbox("Misura", ["Conteggi", "Percentuali"], key=k("measure"))
            orient = st.selectbox("Orientamento", ["Verticale", "Orizzontale"], key=k("orient"))

            ser = df[var_cat]
            freq = ser.value_counts(dropna=False).reset_index(name="count").rename(columns={"index": var_cat})
            freq[var_cat] = freq[var_cat].astype(str)
            freq["percent"] = freq["count"] / len(df) * 100

            if show_mode == "Percentuali":
                y_col = "percent"; y_title = "Percentuale"; textfmt = "%{text:.1f}%"
            else:
                y_col = "count"; y_title = "Frequenza"; textfmt = "%{text}"

            if orient == "Verticale":
                fig_cat = px.bar(freq, x=var_cat, y=y_col, text=y_col, template="simple_white")
                fig_cat.update_traces(texttemplate=textfmt, textposition="outside")
            else:
                fig_cat = px.bar(freq, x=y_col, y=var_cat, text=y_col, orientation="h", template="simple_white")
                fig_cat.update_traces(texttemplate=textfmt, textposition="outside")

            fig_cat.update_layout(title=f"Distribuzione di {var_cat}", height=520, font=dict(size=16))
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Nessuna variabile categoriale selezionata.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 4 · Esportazione + Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 4 · Esportazione e navigazione")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica CSV corrente", data=csv_bytes, file_name="dataset_corrente.csv", mime="text/csv", key=k("download"))

st.markdown("---")
if st.button("➡️ Vai a: Explore Distributions", key=k("go_next"), use_container_width=True):
    st.switch_page("pages/3_📊_Explore_Distributions.py")
