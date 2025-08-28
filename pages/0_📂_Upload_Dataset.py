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

# Stile: pulsanti più bassi (altezza ridotta)
st.markdown("""
<style>
.nav-card {
  border-radius: 14px; padding: 14px; margin-bottom: 16px;
  text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
}
.nav-emoji { font-size: 34px; margin-bottom: 4px; }
.nav-title { margin: 2px 0 2px 0; color: #333; }
.nav-desc { margin: 0 0 8px 0; color: #555; font-size: 0.9em; }
.nav-btn {
  background-color: #4CAF50; color: white;
  padding: 4px 12px;         /* più stretto in altezza */
  line-height: 1.0;          /* riduce l'altezza del bottone */
  border: none; border-radius: 10px; cursor: pointer; font-size: 0.9em;
}
.nav-btn:hover { filter: brightness(0.95); }
</style>
""", unsafe_allow_html=True)

def _exists(rel_path: str) -> str | None:
    """Ritorna il path se esiste, altrimenti None."""
    return rel_path if Path(rel_path).exists() else None

# Card: (path, icon, title, desc, bgcolor)
PAGES = [
    ("pages/1_🧹_Data_Cleaning.py",                  "🧹", "Cleaning",          "Gestione missing values e filtri", "#e6f7ff"),
    ("pages/2_📈_Descriptive_Statistics.py",         "📈", "Descrittive",       "Statistiche di base e riepilogo variabili", "#fff5e6"),
    ("pages/3_📊_Explore_Distributions.py",          "📊", "Distribuzioni",     "Istogrammi, boxplot e violino", "#f9e6ff"),
    ("pages/4_🔍_Assumption_Checks.py",              "🔍", "Assunzioni",        "Verifica normalità e omoscedasticità", "#e6ffe6"),
    ("pages/5_🧪_Statistical_Tests.py",              "🧪", "Test statistici",   "Confronti parametrici e non parametrici", "#fff0f0"),
    ("pages/6_🔗_Correlation_Analysis.py",           "🔗", "Correlazioni",      "Relazioni tra variabili e heatmap", "#f0f5ff"),
    ("pages/7_📂_Subgroup_Analysis.py",              "📂", "Sottogruppi",       "Confronti e descrittive per sottogruppi", "#eef7ff"),
    ("pages/8_🧱_Regression.py",                     "🧱", "Regressione",       "Lineare, Logistica, Poisson", "#e8f5e9"),
    ("pages/9_🧪_Analisi_Test_Diagnostici.py",       "🔬", "Test diagnostici",  "Sens., Spec., LR, ROC/PR, DCA, Calibrazione", "#fff7e6"),
    ("pages/10_📏_Agreement.py",                     "📏", "Agreement",         "Bland–Altman, CCC, Deming, ICC, Kappa", "#e6f0ff"),
    ("pages/11_📈_Analisi_di_Sopravvivenza.py",      "🧭", "Sopravvivenza",     "KM, Nelson–Aalen, Cox PH, AFT", "#f0fff0"),
    ("pages/12_📈_Longitudinale_Misure_Ripetute.py", "📉", "Longitudinale",     "LMM (RI/RS) e GEE, diagnostica", "#f0f8ff"),
    ("pages/13_📘_Glossary.py",                      "📘", "Glossario",         "Termini usati nell’app e definizioni", "#eef5ff"),
]

st.divider()
st.subheader("🚀 Navigazione rapida agli step")

cols = st.columns(2)  # due colonne
i = 0
for page_path, icon, title, desc, color in PAGES:
    page_path = _exists(page_path)
    if page_path is None:
        continue

    card_html = f"""
    <div class="nav-card" style="background-color:{color};">
        <div class="nav-emoji">{icon}</div>
        <h3 class="nav-title">{title}</h3>
        <p class="nav-desc">{desc}</p>
        <a href='/{page_path}' target='_self' style='text-decoration:none;'>
            <button class="nav-btn">Vai →</button>
        </a>
    </div>
    """
    with cols[i % 2]:
        st.markdown(card_html, unsafe_allow_html=True)
    i += 1
