# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert
from core.plots import observed_vs_theoretical_normal, box_violin, qq_plot

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

# Controllo qualità (facoltativo)
with st.expander("🔎 Controllo rapido qualità", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)

# -------------------------------------------------
# Selettori
# -------------------------------------------------
num_cols = list(df.select_dtypes(include="number").columns)
st.subheader("Selezione variabili")
c1, c2 = st.columns([1.2, 1])
with c1:
    target = st.selectbox("Variabile continua", options=num_cols)
with c2:
    use_manual_bins = st.checkbox("Imposta manualmente il numero di classi", value=False)
bins = st.slider("Classi (se impostate manualmente)", 8, 30, 12, 1, disabled=not use_manual_bins)

# -------------------------------------------------
# Header descrittivo (come nello screenshot)
# -------------------------------------------------
st.markdown(f"""
### 📊 **Distribution of** **`{target}`**
> ℹ️ This plot compares the **observed distribution** (histogram + KDE)  
> with the **theoretical normal distribution** (dashed red line).
""")

# -------------------------------------------------
# Grafico principale: Observed vs Theoretical (match screenshot)
# -------------------------------------------------
fig_dist = observed_vs_theoretical_normal(
    df[target],
    bins=(bins if use_manual_bins else None),
    title=f"Observed vs Theoretical Distribution for '{target}'",
    x_label=target
)
st.plotly_chart(fig_dist, use_container_width=True)

# -------------------------------------------------
# Q-Q plot (robusto) e Box/Violin opzionali
# -------------------------------------------------
st.subheader("Q-Q plot (normalità)")
try:
    fig_qq = qq_plot(df[target], title=f"Q-Q plot — {target}")
    st.plotly_chart(fig_qq, use_container_width=True)
except Exception:
    st.info("Q-Q plot non disponibile (dati insufficienti o SciPy assente).")

st.subheader("Box / Violin")
use_violin = st.toggle("Usa violin plot (invece del boxplot)", value=False)
from core.plots import box_violin as _box
fig_box = _box(df[target], by=None, show_violin=use_violin,
               title=("Violin" if use_violin else "Box"))
st.plotly_chart(fig_box, use_container_width=True)

# -------------------------------------------------
# Test di normalità (semplice)
# -------------------------------------------------
st.subheader("Verifica di normalità (semplice)")
def normality_tests(x: pd.Series) -> dict:
    x = x.dropna().astype(float).values
    out = {}
    if x.size < 3:
        return {"Nota": "Campione troppo piccolo"}
    if spstats is None:
        return {"Nota": "SciPy non disponibile"}
    try:
        W, p = spstats.shapiro(x if x.size <= 5000 else x[:5000])
        out["Shapiro-Wilk"] = {"stat": float(W), "pvalue": float(p)}
    except Exception:
        pass
    try:
        K2, p = spstats.normaltest(x)
        out["D’Agostino–Pearson K²"] = {"stat": float(K2), "pvalue": float(p)}
    except Exception:
        pass
    try:
        ad = spstats.anderson(x, dist="norm")
        crit_5 = None
        if hasattr(ad, "significance_level") and 5.0 in ad.significance_level:
            idx = list(ad.significance_level).index(5.0)
            crit_5 = float(ad.critical_values[idx])
        out["Anderson–Darling"] = {"stat": float(ad.statistic), "crit_at_5%": crit_5}
    except Exception:
        pass
    return out

tests = normality_tests(df[target])

if "Nota" in tests:
    st.info(tests["Nota"])
else:
    rows = []
    for name, res in tests.items():
        if "pvalue" in res:
            rows.append({"Test": name, "Statistica": round(res["stat"], 4), "p-value": round(res["pvalue"], 4)})
        else:
            rows.append({"Test": name, "Statistica": round(res["stat"], 4),
                         "Soglia 5%": (None if res["crit_at_5%"] is None else round(res["crit_at_5%"], 4))})
    st.table(pd.DataFrame(rows))

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
        st.error("Conclusione: la variabile **non segue** la normale secondo almeno un test.\n\n" +
                 "\n".join(f"- {v}" for v in verdicts))
    else:
        st.success("Conclusione: nessun test indica deviazioni significative dalla normalità (al 5%).")

# -------------------------------------------------
# Aggiungi al Results Summary
# -------------------------------------------------
st.divider()
if st.button("➕ Aggiungi grafici e risultati al Results Summary"):
    st.session_state.report_items.append({"type": "figure", "title": f"Observed vs Theoretical — {target}", "figure": fig_dist.to_dict()})
    try:
        st.session_state.report_items.append({"type": "figure", "title": f"Q-Q plot — {target}", "figure": fig_qq.to_dict()})
    except NameError:
        pass
    st.session_state.report_items.append({"type": "figure", "title": ("Violin" if use_violin else "Box") + f" — {target}", "figure": fig_box.to_dict()})
    st.success("Elementi aggiunti al Results Summary.")
