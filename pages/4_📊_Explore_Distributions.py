# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert
from core.plots import hist_kde, box_violin, qq_plot

# opzionale: test di normalità
try:
    from scipy import stats as spstats
except Exception:
    spstats = None

init_state()

st.title("📊 Step 3 — Esplora le distribuzioni")

# Requisito dataset
if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = st.session_state.df

# Riepilogo rapido qualità (facoltativo)
with st.expander("🔎 Controllo rapido qualità", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)

# --------------------------
# Selettori
# --------------------------
num_cols = list(df.select_dtypes(include="number").columns)
cat_cols = list(df.select_dtypes(include=["object","category","bool"]).columns)

st.subheader("Selezione variabili")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    target = st.selectbox("Variabile continua", options=num_cols, help="Scegli una variabile numerica da esplorare.")
with c2:
    by_group = st.selectbox("Raggruppa per (opzionale)", options=["— nessuno —"] + cat_cols, index=0,
                            help="Mostra distribuzioni separate per categoria.")
with c3:
    bins = st.slider("Numero di classi (istogramma)", min_value=10, max_value=100, value=30, step=5)

group_series = None if by_group == "— nessuno —" else df[by_group]

# --------------------------
# Grafici
# --------------------------
st.subheader("Grafici principali")

# Istogramma + KDE
fig_hist = hist_kde(df[target], by=group_series, bins=bins, show_kde=(group_series is None),
                    title=f"Istogramma — {target}" + (f" per {by_group}" if group_series is not None else ""))
st.plotly_chart(fig_hist, use_container_width=True)

# Box o Violin
colA, colB = st.columns([1, 1])
with colA:
    use_violin = st.toggle("Usa violin plot (invece del boxplot)", value=False)
fig_box = box_violin(df[target], by=group_series, show_violin=use_violin,
                     title=("Violin" if use_violin else "Box") + (f" per {by_group}" if group_series is not None else ""))
st.plotly_chart(fig_box, use_container_width=True)

# Q-Q plot (normalità)
fig_qq = qq_plot(df[target], title=f"Q-Q plot — {target}")
st.plotly_chart(fig_qq, use_container_width=True)

# --------------------------
# Test di normalità (spiegati)
# --------------------------
st.subheader("Verifica di normalità (semplice)")

with st.expander("Cosa significa normalità? (spiegazione breve)", expanded=False):
    st.markdown("""
La **normalità** indica quanto i dati seguono una **distribuzione normale** (a campana).  
Per molte analisi (es. *t-test* classici) è una **assunzione**: se violata, si preferiscono test **non parametrici**.

- **Shapiro–Wilk** (consigliato, dati piccoli/medi): p-value < 0.05 ⇒ **non** normale.
- **D’Agostino–Pearson (K²)** (alternative su campioni medi/grandi): p-value < 0.05 ⇒ **non** normale.
- **Anderson–Darling**: fornisce un valore critico; se statistica > critico ⇒ **non** normale.
""")

def normality_tests(x: pd.Series) -> dict:
    x = x.dropna().astype(float).values
    out = {}
    if x.size < 3:
        return {"Nota": "Campione troppo piccolo per i test"}
    if spstats is None:
        return {"Nota": "SciPy non disponibile: test non eseguibili"}
    # Shapiro (bene fino a ~5000 osservazioni)
    try:
        W, p = spstats.shapiro(x if x.size <= 5000 else x[:5000])
        out["Shapiro-Wilk"] = {"stat": float(W), "pvalue": float(p)}
    except Exception:
        pass
    # D’Agostino-Pearson
    try:
        K2, p = spstats.normaltest(x)
        out["D’Agostino–Pearson K²"] = {"stat": float(K2), "pvalue": float(p)}
    except Exception:
        pass
    # Anderson–Darling
    try:
        ad = spstats.anderson(x, dist="norm")
        out["Anderson–Darling"] = {
            "stat": float(ad.statistic),
            "crit_at_5%": float(ad.critical_values[list(ad.significance_level).index(5.0)]) if 5.0 in ad.significance_level else None
        }
    except Exception:
        pass
    return out

tests = normality_tests(df[target])

# Presentazione risultati + interpretazione
if "Nota" in tests:
    st.info(tests["Nota"])
else:
    import pandas as pd
    rows = []
    for name, res in tests.items():
        if "pvalue" in res:
            rows.append({"Test": name, "Statistica": round(res["stat"], 4), "p-value": round(res["pvalue"], 4)})
        else:
            rows.append({"Test": name, "Statistica": round(res["stat"], 4), "Soglia 5%": (None if res["crit_at_5%"] is None else round(res["crit_at_5%"],4))})
    st.table(pd.DataFrame(rows))

    # Interpretazione semplice
    verdicts = []
    if "Shapiro-Wilk" in tests and tests["Shapiro-Wilk"]["pvalue"] < 0.05:
        verdicts.append("Shapiro–Wilk: **non** normale (p<0.05).")
    if "D’Agostino–Pearson K²" in tests and tests["D’Agostino–Pearson K²"]["pvalue"] < 0.05:
        verdicts.append("D’Agostino–Pearson: **non** normale (p<0.05).")
    if "Anderson–Darling" in tests:
        crit = tests["Anderson–Darling"]["crit_at_5%"]
        if crit is not None and tests["Anderson–Darling"]["stat"] > crit:
            verdicts.append("Anderson–Darling: **non** normale (stat > soglia 5%).")

    if verdicts:
        st.error("Conclusione: la variabile **non segue** la normale secondo almeno un test.\n\n" + "\n".join(f"- {v}" for v in verdicts))
        st.markdown("""
**Cosa fare?**  
- Usare test **non parametrici** (es. Mann–Whitney, Wilcoxon, Kruskal–Wallis).  
- Valutare trasformazioni (Log10, Box-Cox) **solo** per migliorare la lettura/robustezza.
""")
    else:
        st.success("Conclusione: nessun test indica deviazioni significative dalla normalità (al livello del 5%).")

# --------------------------
# Aggiungi al Results Summary
# --------------------------
st.divider()
if st.button("➕ Aggiungi grafici e risultati al Results Summary"):
    # Nota: per l’esportazione a immagine potremo usare 'kaleido'; per ora salviamo gli oggetti Plotly
    st.session_state.report_items.append({"type": "figure", "title": f"Istogramma — {target}", "figure": fig_hist.to_dict()})
    st.session_state.report_items.append({"type": "figure", "title": ("Violin" if use_violin else "Box") + f" — {target}", "figure": fig_box.to_dict()})
    st.session_state.report_items.append({"type": "figure", "title": f"Q-Q plot — {target}", "figure": fig_qq.to_dict()})
    # Salviamo anche una sintesi normalità
    st.session_state.report_items.append({"type": "text", "title": f"Normalità — {target}", "content": str(tests)})
    st.success("Elementi aggiunti al Results Summary.")
