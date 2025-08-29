
import streamlit as st
from core.state import init_state
from core.io import load_sample
from core.validate import dataset_diagnostics
from core.ui import quality_alert, show_missing_table

st.set_page_config(
    page_title="Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={ "About": "Created by Mathsly Research" }
)

init_state()

st.title("📊 Statistical Analysis — Dashboard iniziale")

# Disclaimer (una-tantum per sessione)
if not st.session_state.accepted_disclaimer:
    st.warning("""
**Disclaimer & Privacy**  
L'analisi è generata automaticamente e **non costituisce consulenza professionale**.  
Verificare sempre i risultati prima di decisioni operative o cliniche.
""")
    if st.checkbox("Ho letto e compreso il Disclaimer & Privacy."):
        st.session_state.accepted_disclaimer = True
        st.success("Grazie. Può procedere.")

# Wizard iniziale se non c'è ancora un dataset caricato
if st.session_state.df is None:
    st.subheader("🚀 Primi passi")
    st.markdown("Carichi un dataset o utilizzi un esempio per esplorare rapidamente le funzionalità.")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### 1) Carica il suo dataset")
        st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a **Upload Dataset**", icon="📂")
    with c2:
        st.markdown("### 2) Prova con un dataset di esempio")
        if st.button("Carica dataset di esempio (clinico)"):
            df = load_sample()
            st.session_state.df = df
            st.session_state.diagnostics = dataset_diagnostics(df)
            st.success("Dataset di esempio caricato.")
    st.divider()
else:
    st.success("✅ Dataset disponibile in memoria.")
    with st.expander("Anteprima dati (prime 10 righe)", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

    st.subheader("📌 Controlli rapidi qualità")
    if st.session_state.diagnostics is None:
        st.session_state.diagnostics = dataset_diagnostics(st.session_state.df)
    diag = st.session_state.diagnostics
    quality_alert(diag)
    with st.expander("Dettaglio missing per colonna"):
        show_missing_table(diag)

    st.markdown("### Prosegua con:")
    st.page_link("pages/2_📈_Descriptive_Statistics.py", label="Descrizione del campione", icon="📈")
    st.page_link("pages/3_📊_Explore_Distributions.py", label="Esplora distribuzioni", icon="📊")
    st.page_link("pages/5_🧪_Statistical_Tests.py", label="Test statistici", icon="🧪")
