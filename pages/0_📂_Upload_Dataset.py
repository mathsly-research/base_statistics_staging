
import streamlit as st
import pandas as pd
from core.state import init_state
from core.io import read_any, load_sample
from core.validate import dataset_diagnostics
from core.ui import quality_alert, show_missing_table

init_state()

st.title("📂 Step 1 — Upload Dataset")

st.markdown("""
Carichi un file **CSV**, **Excel (.xlsx/.xls)** o **Parquet**.
- Ogni riga = un'osservazione (es. paziente)
- Ogni colonna = variabile (es. Età, Sesso, BMI)
- Dimensione massima consigliata: ~200 MB
""")

with st.expander("Suggerimenti sul formato", expanded=False):
    st.markdown("""
- Eviti header duplicati o celle unite in Excel.
- Per CSV, usi separatore **virgola** o **punto e virgola**.
- Le date in formato ISO (es. `2024-03-01`).
""")

c1, c2 = st.columns([2,1])
with c1:
    file = st.file_uploader("Selezioni un file", type=["csv","xlsx","xls","parquet"])
with c2:
    st.markdown("**Oppure**")
    if st.button("Usa dataset di esempio"):
        df_demo = load_sample()
        st.session_state.df = df_demo
        st.session_state.diagnostics = dataset_diagnostics(df_demo)
        st.success("Dataset di esempio caricato.")
        st.experimental_rerun()

delimiter = st.selectbox("Separatore (solo per CSV)", ["Auto", ",", ";", "\t"], index=0)

if file is not None:
    try:
        sep = None if delimiter == "Auto" else ("," if delimiter == "," else ("\t" if delimiter == "\t" else ";"))
        df = read_any(file, delimiter=sep)
        # Pulizia semplice: rimozione colonne totalmente vuote
        df = df.dropna(axis=1, how="all")
        st.session_state.df = df
        st.session_state.diagnostics = dataset_diagnostics(df)
        st.success("✅ Dataset caricato con successo.")
        with st.expander("Anteprima (prime 10 righe)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Righe: {df.shape[0]} • Colonne: {df.shape[1]}")
    except Exception as e:
        st.error(f"Errore in lettura: {e}")

if st.session_state.df is not None:
    st.subheader("🔎 Validazione rapida")
    diag = st.session_state.diagnostics or dataset_diagnostics(st.session_state.df)
    st.session_state.diagnostics = diag
    quality_alert(diag)
    with st.expander("Dettaglio missing per colonna"):
        show_missing_table(diag)

    # Azioni rapide
    st.markdown("### Azioni rapide")
    colA, colB = st.columns(2)
    with colA:
        if diag.get("n_duplicates", 0) > 0 and st.button("Rimuovi righe duplicate"):
            st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
            st.session_state.diagnostics = dataset_diagnostics(st.session_state.df)
            st.success("Righe duplicate rimosse.")
            st.experimental_rerun()
    with colB:
        constant_cols = diag.get("constant_cols", [])
        if constant_cols and st.button("Elimina colonne costanti"):
            st.session_state.df = st.session_state.df.drop(columns=constant_cols, errors="ignore")
            st.session_state.diagnostics = dataset_diagnostics(st.session_state.df)
            st.success("Colonne costanti eliminate.")
            st.experimental_rerun()

    st.divider()
    st.markdown("### Prossimi passi")
    st.page_link("pages/2_📈_Descriptive_Statistics.py", label="Vai a: Descrizione del campione", icon="📈")
    st.page_link("pages/4_📊_Explore_Distributions.py", label="Vai a: Esplora distribuzioni", icon="📊")
    st.page_link("pages/5_🧪_Statistical_Tests.py", label="Vai a: Test statistici", icon="🧪")
