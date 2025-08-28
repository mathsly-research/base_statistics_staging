# pages/3_📊_Explore_Distributions.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except Exception:
    px = None

# ──────────────────────────────────────────────────────────────────────────────
# Data Store centralizzato (+ fallback)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized()
        df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia.")
            st.stop()
        return df
    def stamp_meta():
        ensure_initialized()
        meta = st.session_state["ds_meta"]
        ver = meta.get("version", 0)
        src = meta.get("source") or "-"
        ts = meta.get("updated_at")
        when = "-"
        if ts:
            from datetime import datetime
            when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
# Config pagina + nav
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explore Distributions", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "ed"  # explore distributions
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Explore Distributions")
st.caption("Esplora le distribuzioni **univariate** e le relazioni **bivariate** con opzioni semplici e guidate.")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Variabili numeriche/categoriali
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Aggiornare l'ambiente per visualizzazioni interattive.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_uni_num, tab_uni_cat, tab_bi = st.tabs(["🔢 Univariate · Continue", "🔠 Univariate · Categoriali", "🔗 Bivariate"])

# =============================================================================
# TAB 1 · UNIVARIATE CONTINUE
# =============================================================================
with tab_uni_num:
    st.subheader("Univariate · Variabili continue")
    if not num_vars:
        st.info("Non sono state rilevate variabili numeriche.")
    else:
        x_var = st.selectbox("Variabile continua", options=num_vars, key=k("x_var"))
        color_by = st.selectbox("Colore per sottogruppo (opz.)", options=["(nessuno)"] + cat_vars, key=k("color_num"))
        color_by = None if color_by == "(nessuno)" else color_by
        nbins = st.slider("Numero di bin (istogramma)", 10, 100, 30, 5, key=k("nbins"))

        col_left, col_right = st.columns(2, vertical_alignment="top")

        with col_left:
            fig_h = px.histogram(df, x=x_var, color=color_by, nbins=nbins, template="simple_white")
            fig_h.update_layout(title=f"Istogramma di {x_var}", height=420, font=dict(size=14))
            st.plotly_chart(fig_h, use_container_width=True)

        with col_right:
            style = st.radio("Secondo grafico", ["Boxplot", "Violin"], horizontal=True, key=k("style"))
            if style == "Boxplot":
                fig_b = px.box(df, y=x_var, color=color_by, points="outliers", template="simple_white")
                fig_b.update_layout(title=f"Boxplot di {x_var}", height=420, font=dict(size=14))
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                fig_v = px.violin(df, y=x_var, color=color_by, box=True, points="outliers", template="simple_white")
                fig_v.update_layout(title=f"Violin plot di {x_var}", height=420, font=dict(size=14))
                st.plotly_chart(fig_v, use_container_width=True)

        with st.expander("ℹ️ Come leggere questi grafici"):
            st.markdown("""
            - **Istogramma**: mostra come i valori si distribuiscono lungo la variabile. Barre alte = valori frequenti.  
            - **Boxplot**: evidenzia mediana, quartili e valori anomali (*outlier*).  
            - **Violin plot**: combina boxplot e stima della densità, utile per capire la forma della distribuzione.  
            """)

# =============================================================================
# TAB 2 · UNIVARIATE CATEGORIALI
# =============================================================================
with tab_uni_cat:
    st.subheader("Univariate · Variabili categoriali")
    if not cat_vars:
        st.info("Non sono state rilevate variabili categoriali.")
    else:
        var_cat = st.selectbox("Variabile categoriale", options=cat_vars, key=k("c_var"))
        measure = st.selectbox("Misura", ["Conteggi", "Percentuali"], key=k("c_measure"))
        orient = st.selectbox("Orientamento", ["Verticale", "Orizzontale"], key=k("c_orient"))

        ser = df[var_cat]
        freq = ser.value_counts(dropna=False).reset_index(name="count").rename(columns={"index": var_cat})
        freq[var_cat] = freq[var_cat].astype(str)
        freq["percent"] = freq["count"] / len(df) * 100

        if measure == "Percentuali":
            y_col, y_title, textfmt = "percent", "Percentuale", "%{text:.1f}%"
        else:
            y_col, y_title, textfmt = "count", "Frequenza", "%{text}"

        if orient == "Verticale":
            fig_c = px.bar(freq, x=var_cat, y=y_col, text=y_col, template="simple_white")
            fig_c.update_traces(texttemplate=textfmt, textposition="outside")
        else:
            fig_c = px.bar(freq, x=y_col, y=var_cat, text=y_col, orientation="h", template="simple_white")
            fig_c.update_traces(texttemplate=textfmt, textposition="outside")

        fig_c.update_layout(title=f"Distribuzione di {var_cat}", height=500, font=dict(size=16))
        st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("ℹ️ Come leggere questo grafico"):
            st.markdown("""
            - **Barre verticali/orizzontali**: indicano quanti casi (o % del totale) appartengono a ciascuna categoria.  
            - Barre più lunghe = categoria più frequente.  
            - Utile per confrontare la distribuzione delle modalità.  
            """)

# =============================================================================
# TAB 3 · BIVARIATE
# =============================================================================
with tab_bi:
    st.subheader("Bivariate rapide")
    y_num = st.selectbox("Variabile numerica (Y, opz.)", options=["(nessuna)"] + num_vars, key=k("bi_y"))
    y_num = None if y_num == "(nessuna)" else y_num
    x_cat = st.selectbox("Variabile categoriale (X, opz.)", options=["(nessuno)"] + cat_vars, key=k("bi_xcat"))
    x_cat = None if x_cat == "(nessuno)" else x_cat

    if y_num and x_cat:
        opt = st.radio("Tipo grafico", ["Boxplot", "Violin"], horizontal=True, key=k("bi_type"))
        if opt == "Boxplot":
            fig = px.box(df, x=x_cat, y=y_num, template="simple_white")
        else:
            fig = px.violin(df, x=x_cat, y=y_num, box=True, points="all", template="simple_white")
        fig.update_layout(height=520, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Come leggere questo grafico"):
            st.markdown(f"""
            - Confronta i valori di **{y_num}** tra i gruppi definiti da **{x_cat}**.  
            - Differenze nella posizione delle mediane = differenze tra gruppi.  
            - La larghezza/forma mostra la variabilità all’interno di ciascun gruppo.  
            """)
    else:
        st.info("Selezioni almeno una variabile numerica (Y) e una categoriale (X).")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Statistiche descrittive", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/2_📈_Descriptive_Statistics.py")
with nav2:
    if st.button("➡️ Vai: Assumption Checks", use_container_width=True, key=k("go_next")):
        st.switch_page("pages/4_🔎_Assumption_Checks.py")
