# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert

# -----------------------------
# Init & dataset check
# -----------------------------
init_state()
st.title("🔗 Step 7 — Analisi di correlazione")

if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df: pd.DataFrame = st.session_state.df
num_cols = list(df.select_dtypes(include="number").columns)

if not num_cols:
    st.error("Non ci sono variabili numeriche nel dataset.")
    st.stop()

# -----------------------------
# Scelta modalità
# -----------------------------
mode = st.radio("Seleziona modalità:", ["Due variabili", "Matrice completa"], horizontal=True)

# -----------------------------
# Correlazione a coppia
# -----------------------------
if mode == "Due variabili":
    c1, c2 = st.columns(2)
    with c1:
        var1 = st.selectbox("Variabile 1", options=num_cols)
    with c2:
        var2 = st.selectbox("Variabile 2", options=[c for c in num_cols if c != var1])

    common = df[[var1, var2]].dropna()

    if common.shape[0] < 3:
        st.error("Dati insufficienti per calcolare la correlazione.")
    else:
        pearson_r, pearson_p = stats.pearsonr(common[var1], common[var2])
        spearman_r, spearman_p = stats.spearmanr(common[var1], common[var2])

        st.subheader("Risultati")
        st.write(f"**Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}**")
        st.write(f"**Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}**")

        # Scatterplot con linea di regressione OLS
        fig = px.scatter(common, x=var1, y=var2, trendline="ols",
                         title=f"Scatterplot: {var1} vs {var2} (linea di regressione OLS)")
        st.plotly_chart(fig, use_container_width=True)

        # ➕ Aggiungi al Results Summary
        if st.button("➕ Aggiungi al Results Summary"):
            st.session_state.report_items.append({
                "type": "text",
                "title": f"Correlazione {var1} vs {var2}",
                "content": {
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p
                }
            })
            st.success(f"Correlazione {var1}–{var2} aggiunta al Results Summary.")

# -----------------------------
# Matrice di correlazione
# -----------------------------
else:
    st.subheader("Matrice di correlazione (Pearson)")
    corr = df[num_cols].corr(method="pearson")

    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Heatmap correlazioni (Pearson)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(corr.round(3), use_container_width=True)

    # ➕ Aggiungi matrice al Results Summary
    if st.button("➕ Aggiungi matrice al Results Summary"):
        st.session_state.report_items.append({
            "type": "table",
            "title": "Matrice di correlazione (Pearson)",
            "content": corr.round(3).to_dict()
        })
        st.success("Matrice di correlazione aggiunta al Results Summary.")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Come leggere i risultati"):
    st.markdown("""
**Interpretazione della forza della correlazione (valori assoluti):**  
- 0.00–0.19 → trascurabile  
- 0.20–0.39 → debole  
- 0.40–0.59 → moderata  
- 0.60–0.79 → forte  
- 0.80–1.00 → molto forte  

**Quando usare Pearson vs Spearman:**  
- **Pearson** → relazioni lineari, dati normali.  
- **Spearman** → relazioni monotone, dati ordinali o non normali.  

⚠️ La correlazione non implica causalità. È solo una misura statistica di associazione.
""")
