
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from core.state import init_state
from core.stats import summarize_continuous, summarize_categorical
from core.validate import dataset_diagnostics
from core.ui import quality_alert, show_missing_table

init_state()

st.title("📈 Step 2 — Descriptive Statistics")

if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi un file nella pagina **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = st.session_state.df

with st.expander("🔎 Stato dataset e qualità (rapido)", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)
    with st.expander("Dettaglio missing per colonna", expanded=False):
        show_missing_table(diag)

st.subheader("Selezione variabili")
num_cols_all = list(df.select_dtypes(include="number").columns)
cat_cols_all = list(df.select_dtypes(include=["object","category","bool"]).columns)

c1, c2 = st.columns(2)
with c1:
    num_cols = st.multiselect("Variabili **continue**", num_cols_all, default=num_cols_all)
with c2:
    cat_cols = st.multiselect("Variabili **categoriche**", cat_cols_all, default=cat_cols_all)

st.subheader("Opzioni per le variabili continue")
c3, c4, c5 = st.columns([1,1,1])
with c3:
    exclude_outliers = st.checkbox("Escludi outlier (regola IQR)", value=False, help="Esclude valori fuori da [Q1-1.5*IQR, Q3+1.5*IQR] per ciascuna variabile.")
with c4:
    transform = st.radio("Trasformazione", options=["none","log10","box-cox"], index=0, help="Per Box-Cox servono valori positivi; se necessario viene applicato uno shift automatico.")
with c5:
    show_raw = st.checkbox("Mostra anche versione **grezza**", value=False)

@st.cache_data(show_spinner=False)
def compute_summaries(df_in: pd.DataFrame, num_cols, cat_cols, exclude_outliers, transform):
    cont = summarize_continuous(df_in, cols=num_cols, exclude_outliers=exclude_outliers, transform=transform if transform!="none" else None)
    cat = summarize_categorical(df_in, cols=cat_cols)
    return cont, cat

cont_summary, cat_summary = compute_summaries(df, tuple(num_cols), tuple(cat_cols), exclude_outliers, transform)

st.subheader("Risultati — Continue")
st.dataframe(cont_summary, use_container_width=True)
st.download_button("Scarica CSV (continue)", cont_summary.to_csv(index=False).encode("utf-8"), file_name="descriptive_continuous.csv")

st.subheader("Risultati — Categoriche")
st.dataframe(cat_summary, use_container_width=True)
st.download_button("Scarica CSV (categoriche)", cat_summary.to_csv(index=False).encode("utf-8"), file_name="descriptive_categorical.csv")

# Opzionale: mostra versione grezza per confronto
if show_raw and transform != "none":
    st.markdown("#### Confronto (senza trasformazioni, senza esclusione outlier)")
    cont_raw, _ = compute_summaries(df, tuple(num_cols), tuple(cat_cols), False, "none")
    st.dataframe(cont_raw, use_container_width=True)

# Aggiungi al report
st.divider()
if st.button("➕ Aggiungi queste tabelle al Results Summary"):
    st.session_state.report_items.append({
        "type": "table",
        "title": "Descriptive — Continuous",
        "data": cont_summary.to_dict(orient="records")
    })
    st.session_state.report_items.append({
        "type": "table",
        "title": "Descriptive — Categorical",
        "data": cat_summary.to_dict(orient="records")
    })
    st.success("Tabelle aggiunte al Results Summary.")

st.info("Suggerimento: passi alla sezione **Explore Distributions** per visualizzare grafici per singola variabile.")
