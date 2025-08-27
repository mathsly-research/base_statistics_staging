# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert
from core.tests import decide_and_test, TestResult

# -----------------------------
# Inizializzazione & check dataset
# -----------------------------
init_state()
st.title("🧪 Step 4 — Test statistici guidati")

if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df: pd.DataFrame = st.session_state.df

with st.expander("🔎 Controllo rapido qualità", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)

# -----------------------------
# Selezione variabili
# -----------------------------
st.subheader("Selezione variabili")
num_cols = list(df.select_dtypes(include="number").columns)
cat_cols = list(df.select_dtypes(exclude="number").columns)
all_cols = list(df.columns)

c1, c2 = st.columns(2)
with c1:
    y = st.selectbox("Variabile di interesse (outcome)", options=all_cols, help="Variabile continua o categorica.")
with c2:
    group = st.selectbox("Raggruppamento (gruppi)", options=[None] + all_cols, format_func=lambda x: "— nessuno —" if x is None else x)

st.caption("Suggerimento: per confronti tra gruppi serve selezionare una variabile di **raggruppamento**.")

# -----------------------------
# Esecuzione (automatica)
# -----------------------------
st.subheader("Risultati")
res = decide_and_test(df, y, group)

if res.get("note"):
    st.info(res["note"])
else:
    design = res.get("design", {})
    if design:
        if design["y_type"] == "continuous":
            st.write(f"**Disegno:** variabile continua, confronto tra gruppi {design['groups']}.")
        elif design["y_type"] == "binary":
            st.write("**Disegno:** variabile binaria, confronto tra distribuzioni (chi-quadrato) e, se binario, due proporzioni.")
        else:
            st.write("**Disegno:** variabile categorica multinomiale, test di indipendenza (chi-quadrato).")

    for i, tr in enumerate(res["tests"], start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {tr.test_name}**")
            if tr.stat is not None:
                st.write(f"Statistica = `{tr.stat:.4f}`")
            if tr.df is not None:
                st.write(f"Gradi di libertà = `{tr.df:.2f}`")
            if tr.pvalue is not None:
                st.write(f"p-value = `{tr.pvalue:.4g}`")

            if tr.estimate_name and tr.estimate_value is not None:
                if tr.estimate_ci:
                    lo, hi = tr.estimate_ci
                    st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}**  (CI 95% [{lo:.4g}, {hi:.4g}])")
                else:
                    st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}**")

            if tr.effect_name and tr.effect_value is not None:
                if tr.effect_ci:
                    elo, ehi = tr.effect_ci
                    st.write(f"{tr.effect_name}: **{tr.effect_value:.4g}**  (CI 95% [{elo:.4g}, {ehi:.4g}])")
                else:
                    st.write(f"{tr.effect_name}: **{tr.effect_value:.4g}**")

            if tr.details and "table" in tr.details:
                st.markdown("**Tabella di contingenza**")
                st.dataframe(tr.details["table"], use_container_width=True)

            if tr.note:
                st.info(tr.note)

    # Spiegazioni semplici
    with st.expander("ℹ️ Come interpretare i risultati (guida rapida)", expanded=False):
        st.markdown("""
- **p-value < 0.05**: differenza/associazione **statisticamente significativa** (al 5%).  
- **Effect size** (es. Hedges g, Cliff’s δ, Cramer’s V): misura **quanto è grande** l’effetto (non solo se è significativo).  
- **CI 95%**: intervallo plausibile per la stima (es. differenza di medie o proporzioni).  
- **Scelte dei test**:
  - Continua + 2 gruppi → *Welch t-test* (robusto a varianze disuguali) e **Mann–Whitney** (non parametrico).
  - Continua + >2 gruppi → **ANOVA** e **Kruskal–Wallis** (non parametrico).
  - Categorica vs raggruppamento → **Chi-quadrato**; se entrambe binarie, anche **confronto due proporzioni**.
""")

# -----------------------------
# Add to Results Summary
# -----------------------------
st.divider()
if st.button("➕ Aggiungi risultati al Results Summary"):
    items = []
    for tr in res.get("tests", []):
        entry = {
            "type": "text",
            "title": tr.test_name,
            "content": {
                "stat": tr.stat,
                "pvalue": tr.pvalue,
                "df": tr.df,
                "estimate_name": tr.estimate_name,
                "estimate_value": tr.estimate_value,
                "estimate_ci": tr.estimate_ci,
                "effect_name": tr.effect_name,
                "effect_value": tr.effect_value,
                "effect_ci": tr.effect_ci,
            }
        }
        items.append(entry)
    st.session_state.report_items.extend(items)
    st.success("Risultati aggiunti al Results Summary.")
